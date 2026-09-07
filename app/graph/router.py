from app.schemas.chat import RouteDecision, RouteName


RAG_HINTS = (
    "文档",
    "知识库",
    "资料",
    "制度",
    "政策",
    "报销",
    "审批",
    "项目",
    "架构",
    "学习路线",
    "rag",
    "redis",
    "langgraph",
)
TOOL_ACTION_HINTS = ("计算", "天气", "查一下", "统计", "算一下")
TOOL_HINTS = TOOL_ACTION_HINTS + ("工具",)
ROUTE_TOOL_HINTS = ("路线规划", "怎么走", "从")
PLAN_HINTS = (
    "制定",
    "规划",
    "拆解",
    "方案",
    "计划",
    "步骤",
    "安排",
    "综合",
    "结合",
    "输出一份",
    "生成一份",
)
FALLBACK_HINTS = ("破解", "攻击", "违法", "绕过鉴权")


class SupervisorRouter:
    """Deterministic supervisor with an auditable routing policy."""

    def decide(self, message: str) -> RouteDecision:
        normalized = message.lower()

        if any(hint in normalized for hint in FALLBACK_HINTS):
            return RouteDecision(
                route=RouteName.fallback,
                confidence=0.92,
                reason="请求命中了安全边界或兜底规则。",
            )

        if _looks_like_tool_action(normalized):
            return RouteDecision(
                route=RouteName.tool,
                confidence=0.82,
                reason="请求包含明确工具动作，需要结构化工具处理。",
            )

        if _looks_like_complex_planning(normalized):
            return RouteDecision(
                route=RouteName.plan,
                confidence=0.74,
                reason="请求需要拆解目标、组织步骤或综合多个能力后再回答。",
            )

        if any(hint in normalized for hint in RAG_HINTS):
            return RouteDecision(
                route=RouteName.rag,
                confidence=0.76,
                reason="请求需要检索知识库后再回答。",
            )

        if any(hint in normalized for hint in TOOL_HINTS) or _looks_like_route_planning(normalized):
            return RouteDecision(
                route=RouteName.tool,
                confidence=0.78,
                reason="请求需要结构化工具处理。",
            )

        return RouteDecision(
            route=RouteName.chat,
            confidence=0.64,
            reason="请求可由普通对话分支处理。",
        )


def _looks_like_route_planning(message: str) -> bool:
    return any(hint in message for hint in ROUTE_TOOL_HINTS) and "到" in message


def _looks_like_tool_action(message: str) -> bool:
    return any(hint in message for hint in TOOL_ACTION_HINTS) or bool(
        __import__("re").search(r"\d+\s*[+\-*/]\s*\d+", message)
    )


def _looks_like_complex_planning(message: str) -> bool:
    if "是什么" in message or "多少" in message:
        return False

    has_planning_hint = any(hint in message for hint in PLAN_HINTS)
    has_task_intent = any(prefix in message for prefix in ("帮我", "请", "给我", "我要", "我想"))
    has_business_context = any(
        hint in message
        for hint in (
            "项目",
            "学习",
            "业务",
            "系统",
            "制度",
            "知识库",
            "审批",
            "报销",
            "部署",
            "上线",
            "agent",
            "rag",
            "langgraph",
        )
    )
    return has_planning_hint and has_task_intent and has_business_context
