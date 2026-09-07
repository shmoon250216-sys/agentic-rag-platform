import json
from pathlib import Path

import aiosqlite

from app.core.config import get_settings
from app.rag.embedding import EmbeddingModel, cosine_similarity, get_embedding_model, tokenize
from app.schemas.memory import MemoryItem, MemoryType


class SQLiteLongTermMemoryStore:
    """SQLite-backed long-term memory store scoped by user."""

    def __init__(
        self,
        database_url: str | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.database_path = _sqlite_path(database_url or get_settings().sqlite_url)
        self.embedding_model = embedding_model or get_embedding_model(get_settings())
        self._initialized = False

    async def add(
        self,
        user_id: str,
        memory_type: MemoryType,
        content: str,
        importance: float = 0.7,
    ) -> MemoryItem:
        await self._ensure_schema()
        normalized_content = " ".join(content.strip().split())
        embedding_json = _embedding_to_json(self.embedding_model.embed(normalized_content))
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute(
                """
                INSERT INTO user_memories (
                    user_id,
                    memory_type,
                    content,
                    importance,
                    embedding_json
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, memory_type, content)
                DO UPDATE SET
                    importance = MAX(user_memories.importance, excluded.importance),
                    embedding_json = excluded.embedding_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, memory_type.value, normalized_content, importance, embedding_json),
            )
            await db.commit()
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, memory_type, content, importance, created_at, updated_at
                FROM user_memories
                WHERE user_id = ? AND memory_type = ? AND content = ?
                """,
                (user_id, memory_type.value, normalized_content),
            )
            row = await cursor.fetchone()
            return _row_to_memory(row)

    async def list_for_user(self, user_id: str, limit: int = 50) -> list[MemoryItem]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT memory_id, user_id, memory_type, content, importance, created_at, updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            return [_row_to_memory(row) for row in await cursor.fetchall()]

    async def search(self, user_id: str, query: str, limit: int = 5) -> list[MemoryItem]:
        await self._ensure_schema()
        query_terms = tokenize(query)
        query_embedding = self.embedding_model.embed(query)

        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT
                    memory_id,
                    user_id,
                    memory_type,
                    content,
                    importance,
                    embedding_json,
                    created_at,
                    updated_at
                FROM user_memories
                WHERE user_id = ?
                ORDER BY importance DESC, updated_at DESC
                LIMIT 100
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()

        scored: list[tuple[float, MemoryItem]] = []
        for row in rows:
            memory = _row_to_memory(row)
            memory_embedding = _embedding_from_row(row)
            if memory_embedding is None:
                memory_embedding = self.embedding_model.embed(memory.content)
                await self._backfill_embedding(memory.memory_id, memory_embedding)

            memory_terms = tokenize(memory.content)
            overlap = query_terms & memory_terms
            lexical_score = len(overlap) / max(len(query_terms), 1) if query_terms else 0
            vector_score = max(cosine_similarity(query_embedding, memory_embedding), 0.0)
            score = _hybrid_memory_score(vector_score, lexical_score, memory.importance)
            if vector_score > 0 or lexical_score > 0 or memory.importance >= 0.85:
                scored.append((score, memory))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [memory for _, memory in scored[:limit]]

    async def delete(self, memory_id: int, user_id: str | None = None) -> bool:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            if user_id:
                cursor = await db.execute(
                    "DELETE FROM user_memories WHERE memory_id = ? AND user_id = ?",
                    (memory_id, user_id),
                )
            else:
                cursor = await db.execute(
                    "DELETE FROM user_memories WHERE memory_id = ?",
                    (memory_id,),
                )
            await db.commit()
            return cursor.rowcount > 0

    async def _backfill_embedding(self, memory_id: int, embedding: list[float]) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                UPDATE user_memories
                SET embedding_json = ?
                WHERE memory_id = ?
                """,
                (_embedding_to_json(embedding), memory_id),
            )
            await db.commit()

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.7,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, memory_type, content)
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_user_memories_user_type
                ON user_memories (user_id, memory_type, updated_at DESC)
                """
            )
            await _ensure_embedding_column(db)
            await db.commit()
        self._initialized = True


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        path = database_url.removeprefix(prefix)
    else:
        path = database_url
    return Path(path)


def _row_to_memory(row: aiosqlite.Row) -> MemoryItem:
    return MemoryItem(
        memory_id=row["memory_id"],
        user_id=row["user_id"],
        memory_type=MemoryType(row["memory_type"]),
        content=row["content"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def _ensure_embedding_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(user_memories)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "embedding_json" not in columns:
        await db.execute("ALTER TABLE user_memories ADD COLUMN embedding_json TEXT")


def _embedding_to_json(embedding: list[float]) -> str:
    return json.dumps(embedding, separators=(",", ":"))


def _embedding_from_row(row: aiosqlite.Row) -> list[float] | None:
    raw_value = row["embedding_json"]
    if not raw_value:
        return None
    if isinstance(raw_value, bytes):
        raw_value = raw_value.decode("utf-8")
    return [float(value) for value in json.loads(raw_value)]


def _hybrid_memory_score(vector_score: float, lexical_score: float, importance: float) -> float:
    return 0.6 * vector_score + 0.25 * lexical_score + 0.15 * importance
