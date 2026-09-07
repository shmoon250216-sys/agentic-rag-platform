import re
from dataclasses import dataclass

from app.memory.long_term_store import SQLiteLongTermMemoryStore
from app.schemas.memory import MemoryItem, MemoryType


SENSITIVE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"api[-_ ]?key", re.IGNORECASE),
    re.compile(r"password|passwd|secret|token", re.IGNORECASE),
    re.compile(r"密码|密钥|令牌|身份证|银行卡"),
)


@dataclass(frozen=True)
class MemoryCandidate:
    memory_type: MemoryType
    content: str
    importance: float


class LongTermMemoryManager:
    """Rule-based memory policy for deciding what should persist across sessions."""

    def __init__(self, store: SQLiteLongTermMemoryStore | None = None) -> None:
        self.store = store or SQLiteLongTermMemoryStore()

    async def remember_from_message(self, user_id: str, message: str) -> list[MemoryItem]:
        candidates = self.extract_candidates(message)
        saved: list[MemoryItem] = []
        for candidate in candidates:
            saved.append(
                await self.store.add(
                    user_id=user_id,
                    memory_type=candidate.memory_type,
                    content=candidate.content,
                    importance=candidate.importance,
                )
            )
        return saved

    async def relevant_memories(self, user_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        return await self.store.search(user_id=user_id, query=query, limit=limit)

    def is_safe_to_store(self, text: str) -> bool:
        return not _contains_sensitive_text(text)

    def extract_candidates(self, message: str) -> list[MemoryCandidate]:
        normalized = " ".join(message.strip().split())
        if len(normalized) < 8 or _contains_sensitive_text(normalized):
            return []

        candidates: list[MemoryCandidate] = []
        if _has_any(normalized, ("我是", "我是一名", "我叫", "我目前", "我的背景")):
            candidates.append(
                MemoryCandidate(MemoryType.profile, f"用户自我描述：{normalized}", 0.85)
            )

        if _has_any(normalized, ("我喜欢", "我希望", "以后请", "尽量", "用中文", "详细", "简洁")):
            candidates.append(
                MemoryCandidate(MemoryType.preference, f"用户偏好：{normalized}", 0.8)
            )

        if _has_goal_signal(normalized):
            candidates.append(
                MemoryCandidate(MemoryType.goal, f"用户长期目标：{normalized}", 0.9)
            )

        if _has_any(normalized, ("我的项目", "这个系统", "当前系统", "项目使用", "技术栈")):
            candidates.append(
                MemoryCandidate(MemoryType.project_fact, f"项目事实：{normalized}", 0.75)
            )

        if _has_any(normalized, ("决定使用", "选择使用", "后续采用", "暂时保持", "不再使用")):
            candidates.append(
                MemoryCandidate(MemoryType.decision, f"关键决策：{normalized}", 0.8)
            )

        return _dedupe_candidates(candidates)


def _contains_sensitive_text(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _has_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _has_goal_signal(text: str) -> bool:
    return _has_any(text, ("我的目标", "我想", "我准备", "我要", "计划")) and _has_any(
        text,
        ("制度", "知识库", "报销", "审批", "部署", "上线", "项目", "学习"),
    )


def _dedupe_candidates(candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
    seen: set[tuple[MemoryType, str]] = set()
    deduped: list[MemoryCandidate] = []
    for candidate in candidates:
        key = (candidate.memory_type, candidate.content)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped
