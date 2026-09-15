from collections.abc import AsyncIterator
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.memory.context import ConversationContext, format_context
from app.llm.client import LLMClient, get_llm_client
from app.rag.retriever import KnowledgeBaseRetriever
from app.schemas.chat import ChatRequest, ChatResponse, RouteName
from app.tools.registry import ToolRegistry

from .router import SupervisorRouter
from .state import AgentState


class AgentWorkflow:
    """LangGraph supervisor workflow for adaptive routing."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.router = SupervisorRouter()
        self.retriever = KnowledgeBaseRetriever()
        self.tools = ToolRegistry()
        self.llm = llm or get_llm_client(get_settings())
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    async def run(
        self,
        request: ChatRequest,
        session_id: str,
        memories: list[str] | None = None,
        context: ConversationContext | None = None,
    ) -> ChatResponse:
        initial_state: AgentState = {
            "user_id": request.user_id,
            "session_id": session_id,
            "message": request.message,
            "query": context.query if context else request.message,
            "history": context.history if context else [],
            "summary": context.summary if context else "",
            "memories": memories or [],
            "sources": [],
            "tool_result": None,
            "error": None,
        }
        try:
            result = await self.graph.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": session_id}},
            )
        finally:
            # History is explicitly rebuilt from SQLite. In-run checkpoints need
            # not accumulate indefinitely or carry fields into the next turn.
            await self.checkpointer.adelete_thread(session_id)
        decision = result["route"]

        return ChatResponse(
            session_id=session_id,
            answer=result.get("answer", "本次请求没有生成回答。"),
            route=decision,
            sources=result.get("sources", []),
            tool_result=result.get("tool_result"),
        )

    async def stream(
        self,
        request: ChatRequest,
        session_id: str,
        memories: list[str] | None = None,
        context: ConversationContext | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        state: AgentState = {
            "user_id": request.user_id,
            "session_id": session_id,
            "message": request.message,
            "query": context.query if context else request.message,
            "history": context.history if context else [],
            "summary": context.summary if context else "",
            "memories": memories or [],
            "sources": [],
            "tool_result": None,
            "error": None,
        }
        decision = self._decide(state)
        state["route"] = decision
        yield "route", decision

        if decision.route == RouteName.rag:
            sources = await self.retriever.search(state.get("query", state["message"]))
            state["sources"] = sources
            answer_parts: list[str] = []
            async for token in self.llm.stream_answer_with_context(
                _with_memory_context(state),
                sources,
            ):
                answer_parts.append(token)
                yield "token", token
            state["answer"] = "".join(answer_parts)
        elif decision.route == RouteName.tool:
            tool_result = await self.tools.run_best_effort(state["message"])
            state["tool_result"] = tool_result
            answer_parts = []
            async for token in self.llm.stream_answer_with_tool_result(
                _with_memory_context(state),
                tool_result,
            ):
                answer_parts.append(token)
                yield "token", token
            state["answer"] = "".join(answer_parts)
        elif decision.route == RouteName.plan:
            answer_parts = []
            async for token in self.llm.stream_chat(_planning_prompt(state)):
                answer_parts.append(token)
                yield "token", token
            state["answer"] = "".join(answer_parts)
        elif decision.route == RouteName.chat:
            answer_parts = []
            async for token in self.llm.stream_chat(_with_memory_context(state)):
                answer_parts.append(token)
                yield "token", token
            state["answer"] = "".join(answer_parts)
        else:
            state["answer"] = (
                "这个请求触发了兜底分支。后续会在这里加入更完整的安全边界、错误恢复和人工提示。"
            )
            yield "token", state["answer"]

        yield (
            "done",
            ChatResponse(
                session_id=session_id,
                answer=state.get("answer", "本次请求没有生成回答。"),
                route=decision,
                sources=state.get("sources", []),
                tool_result=state.get("tool_result"),
            ),
        )

    def _build_graph(self) -> Any:
        builder = StateGraph(AgentState)
        builder.add_node("supervisor", self._supervisor_node)
        builder.add_node("rag", self._rag_node)
        builder.add_node("tool", self._tool_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("chat", self._chat_node)
        builder.add_node("fallback", self._fallback_node)

        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges(
            "supervisor",
            self._select_branch,
            {
                RouteName.rag.value: "rag",
                RouteName.tool.value: "tool",
                RouteName.plan.value: "plan",
                RouteName.chat.value: "chat",
                RouteName.fallback.value: "fallback",
            },
        )
        builder.add_edge("rag", END)
        builder.add_edge("tool", END)
        builder.add_edge("plan", END)
        builder.add_edge("chat", END)
        builder.add_edge("fallback", END)
        return builder.compile(checkpointer=self.checkpointer)

    def _decide(self, state):
        # Current explicit intent/safety wins; only enrich underspecified follow-ups.
        current = self.router.decide(state["message"])
        if current.route != RouteName.chat:
            return current
        contextual = self.router.decide(state.get("query", state["message"]))
        # Historical tool/safety keywords must not execute an old action.
        return contextual if contextual.route == RouteName.rag else current

    async def _supervisor_node(self, state: AgentState) -> AgentState:
        return {"route": self._decide(state)}

    def _select_branch(self, state: AgentState) -> str:
        decision = state.get("route")
        if decision is None:
            return RouteName.fallback.value
        return decision.route.value

    async def _rag_node(self, state: AgentState) -> AgentState:
        sources = await self.retriever.search(state.get("query", state["message"]))
        answer = await self.llm.answer_with_context(_with_memory_context(state), sources)
        return {"sources": sources, "answer": answer}

    async def _tool_node(self, state: AgentState) -> AgentState:
        tool_result = await self.tools.run_best_effort(state["message"])
        answer = await self.llm.answer_with_tool_result(_with_memory_context(state), tool_result)
        return {"tool_result": tool_result, "answer": answer}

    async def _plan_node(self, state: AgentState) -> AgentState:
        return {"answer": await self.llm.chat(_planning_prompt(state))}

    async def _chat_node(self, state: AgentState) -> AgentState:
        return {"answer": await self.llm.chat(_with_memory_context(state))}

    async def _fallback_node(self, state: AgentState) -> AgentState:
        return {
            "answer": (
                "这个请求触发了兜底分支。后续会在这里加入更完整的安全边界、错误恢复和人工提示。"
            )
        }


def _with_memory_context(state: AgentState) -> str:
    return format_context(state)


def _planning_prompt(state: AgentState) -> str:
    return (
        "你正在复杂任务规划分支。请先明确用户目标，再给出分步骤方案、"
        "需要用到的系统能力、风险点和下一步执行建议。"
        "如果需要知识库、工具或长期记忆，请说明应该如何调用它们。\n\n"
        f"{_with_memory_context(state)}"
    )
