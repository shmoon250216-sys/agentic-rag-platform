from app.graph.router import SupervisorRouter
from app.schemas.chat import RouteName


def test_router_selects_rag_for_knowledge_base_question() -> None:
    decision = SupervisorRouter().decide("请介绍一下这个项目的架构")
    assert decision.route == RouteName.rag


def test_router_selects_rag_for_learning_plan_question() -> None:
    decision = SupervisorRouter().decide("这个 Agent 项目的学习路线是什么")
    assert decision.route == RouteName.rag


def test_router_selects_tool_for_calculation_question() -> None:
    decision = SupervisorRouter().decide("帮我计算 12 + 30")
    assert decision.route == RouteName.tool


def test_router_selects_tool_for_doc_stats_question() -> None:
    decision = SupervisorRouter().decide("统计一下知识库有多少文档")
    assert decision.route == RouteName.tool


def test_router_selects_plan_for_complex_project_task() -> None:
    decision = SupervisorRouter().decide("请帮我制定一个把新报销制度接入知识库的方案")
    assert decision.route == RouteName.plan


def test_router_selects_chat_for_general_question() -> None:
    decision = SupervisorRouter().decide("你好，今天适合学习什么")
    assert decision.route == RouteName.chat
