from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class MessageRecord:
    role: str
    content: str


@dataclass
class SessionRecord:
    session_id: str
    user_id: str
    messages: list[MessageRecord] = field(default_factory=list)


class InMemorySessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}

    def get_or_create(self, user_id: str, session_id: str | None = None) -> SessionRecord:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        new_session = SessionRecord(session_id=session_id or str(uuid4()), user_id=user_id)
        self._sessions[new_session.session_id] = new_session
        return new_session

    def append(self, session_id: str, role: str, content: str) -> None:
        self._sessions[session_id].messages.append(MessageRecord(role=role, content=content))

