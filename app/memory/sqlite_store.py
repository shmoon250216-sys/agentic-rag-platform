import re
from pathlib import Path
from uuid import uuid4

import aiosqlite

from app.core.config import get_settings
from app.core.errors import AppError
from app.memory.context import compact_summary
from app.memory.store import MessageRecord, SessionRecord
from app.schemas.session import MessageItem, SessionSummary


class SQLiteSessionStore:
    """SQLite-backed conversation store for local durable memory."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_path = _sqlite_path(database_url or get_settings().sqlite_url)
        self._initialized = False

    async def get_or_create(self, user_id: str, session_id: str | None = None) -> SessionRecord:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            if session_id:
                existing = await self._fetch_session(db, session_id)
                if existing:
                    if existing.user_id != user_id:
                        raise AppError(
                            code="SESSION_ACCESS_DENIED",
                            message="Session belongs to another user",
                            status_code=403,
                        )
                    return existing

            new_session_id = session_id or str(uuid4())
            await db.execute(
                """
                INSERT INTO sessions (session_id, user_id)
                VALUES (?, ?)
                """,
                (new_session_id, user_id),
            )
            await db.commit()
            return SessionRecord(session_id=new_session_id, user_id=user_id)

    async def append(self, session_id: str, role: str, content: str) -> None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT INTO messages (session_id, role, content)
                VALUES (?, ?, ?)
                """,
                (session_id, role, content),
            )
            if role == "user":
                await db.execute(
                    """
                    UPDATE sessions
                    SET
                        title = CASE
                            WHEN title IS NULL OR title = '' THEN ?
                            ELSE title
                        END,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    (_generate_session_title(content), session_id),
                )
            else:
                await db.execute(
                    """
                    UPDATE sessions
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = ?
                    """,
                    (session_id,),
                )
            cursor = await db.execute(
                """SELECT role, content FROM messages WHERE session_id = ?
                   ORDER BY message_id DESC LIMIT -1 OFFSET ?""",
                (session_id, get_settings().session_max_messages),
            )
            removed = list(reversed(await cursor.fetchall()))
            if removed:
                cursor = await db.execute(
                    "SELECT context_summary FROM sessions WHERE session_id = ?", (session_id,)
                )
                row = await cursor.fetchone()
                digest = compact_summary(
                    row[0] or "",
                    [r[1] for r in removed if r[0] == "user"],
                    get_settings().context_summary_chars,
                )
                await db.execute(
                    "UPDATE sessions SET context_summary = ? WHERE session_id = ?",
                    (digest, session_id),
                )
            await db.execute(
                """DELETE FROM messages WHERE session_id = ? AND message_id NOT IN (
                    SELECT message_id FROM messages WHERE session_id = ?
                    ORDER BY message_id DESC LIMIT ?
                )""",
                (session_id, session_id, get_settings().session_max_messages),
            )
            await db.commit()

    async def list_sessions(
        self, user_id: str | None = None, limit: int = 20
    ) -> list[SessionSummary]:
        await self._ensure_schema()
        params: tuple[object, ...]
        where_clause = ""
        if user_id:
            where_clause = "WHERE s.user_id = ?"
            params = (user_id, limit)
        else:
            params = (limit,)

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT
                    s.session_id,
                    s.user_id,
                    s.title,
                    s.created_at,
                    s.updated_at,
                    COUNT(m.message_id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                {where_clause}
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                params,
            )
            rows = await cursor.fetchall()
            return [
                SessionSummary(
                    session_id=row["session_id"],
                    user_id=row["user_id"],
                    title=row["title"] or "新会话",
                    message_count=row["message_count"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    async def list_messages(self, session_id: str) -> list[MessageItem]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT message_id, session_id, role, content, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY message_id ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [
                MessageItem(
                    message_id=row["message_id"],
                    session_id=row["session_id"],
                    role=row["role"],
                    content=row["content"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def _fetch_session(
        self,
        db: aiosqlite.Connection,
        session_id: str,
    ) -> SessionRecord | None:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT session_id, user_id, context_summary
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None

        message_cursor = await db.execute(
            """
            SELECT role, content
            FROM messages
            WHERE session_id = ?
            ORDER BY message_id ASC
            """,
            (session_id,),
        )
        messages = [
            MessageRecord(role=message["role"], content=message["content"])
            for message in await message_cursor.fetchall()
        ]
        return SessionRecord(
            session_id=row["session_id"],
            user_id=row["user_id"],
            messages=messages,
            summary=row["context_summary"] or "",
        )

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor = await db.execute("PRAGMA table_info(sessions)")
            if "context_summary" not in {row[1] for row in await cursor.fetchall()}:
                await db.execute(
                    "ALTER TABLE sessions ADD COLUMN context_summary TEXT NOT NULL DEFAULT ''"
                )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages (session_id, message_id)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_updated
                ON sessions (user_id, updated_at DESC)
                """
            )
            await self._ensure_title_column(db)
            await db.commit()
        self._initialized = True

    async def _ensure_title_column(self, db: aiosqlite.Connection) -> None:
        cursor = await db.execute("PRAGMA table_info(sessions)")
        columns = {row[1] for row in await cursor.fetchall()}
        if "title" not in columns:
            await db.execute("ALTER TABLE sessions ADD COLUMN title TEXT")

        cursor = await db.execute(
            """
            SELECT
                s.session_id,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.session_id = s.session_id AND m.role = 'user'
                    ORDER BY m.message_id ASC
                    LIMIT 1
                ) AS first_user_message
            FROM sessions s
            WHERE s.title IS NULL OR s.title = ''
            """
        )
        updates = [
            (_generate_session_title(row[1]), row[0]) for row in await cursor.fetchall() if row[1]
        ]
        if updates:
            await db.executemany(
                "UPDATE sessions SET title = ? WHERE session_id = ?",
                updates,
            )


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        path = database_url.removeprefix(prefix)
    else:
        path = database_url
    return Path(path)


def _generate_session_title(message: str, max_length: int = 24) -> str:
    normalized = re.sub(r"\s+", " ", message).strip()
    normalized = re.sub(
        r"^(?:你好|您好|好的|好|请问|麻烦你|麻烦|请你|请|能不能|可以不可以)[，,。.!！?？\s]*",
        "",
        normalized,
    )
    normalized = re.sub(
        r"^(?:我想知道|我想了解|我想问|我希望|我需要|帮我|给我|现在帮我|现在开始)[，,。.!！?？\s]*",
        "",
        normalized,
    )
    normalized = normalized.strip("，,。.!！?？；;：:、 ") or "新会话"
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length].rstrip('，,。.!！?？；;：:、 ')}…"
