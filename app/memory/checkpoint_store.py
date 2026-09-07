import json
from pathlib import Path
from typing import Any

import aiosqlite
from pydantic import BaseModel

from app.core.config import get_settings
from app.schemas.checkpoint import GraphCheckpointDetail, GraphCheckpointSummary


class SQLiteGraphCheckpointStore:
    """Persist final LangGraph run snapshots for inspection and replay planning."""

    def __init__(self, database_url: str | None = None) -> None:
        self.database_path = _sqlite_path(database_url or get_settings().sqlite_url)
        self._initialized = False

    async def append(
        self,
        session_id: str,
        route: str,
        state: dict[str, Any],
    ) -> int:
        await self._ensure_schema()
        state_json = json.dumps(state, ensure_ascii=False, default=_json_default)
        async with aiosqlite.connect(self.database_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO graph_checkpoints (session_id, route, state_json)
                VALUES (?, ?, ?)
                """,
                (session_id, route, state_json),
            )
            await db.commit()
            return int(cursor.lastrowid)

    async def list_for_session(self, session_id: str) -> list[GraphCheckpointSummary]:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT checkpoint_id, session_id, route, created_at
                FROM graph_checkpoints
                WHERE session_id = ?
                ORDER BY checkpoint_id ASC
                """,
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [
                GraphCheckpointSummary(
                    checkpoint_id=row["checkpoint_id"],
                    session_id=row["session_id"],
                    route=row["route"],
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    async def get(self, checkpoint_id: int) -> GraphCheckpointDetail | None:
        await self._ensure_schema()
        async with aiosqlite.connect(self.database_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT checkpoint_id, session_id, route, state_json, created_at
                FROM graph_checkpoints
                WHERE checkpoint_id = ?
                """,
                (checkpoint_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return GraphCheckpointDetail(
                checkpoint_id=row["checkpoint_id"],
                session_id=row["session_id"],
                route=row["route"],
                created_at=row["created_at"],
                state=json.loads(row["state_json"]),
            )

    async def _ensure_schema(self) -> None:
        if self._initialized:
            return

        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_checkpoints (
                    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    route TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_graph_checkpoints_session
                ON graph_checkpoints (session_id, checkpoint_id)
                """
            )
            await db.commit()
        self._initialized = True


def _sqlite_path(database_url: str) -> Path:
    prefix = "sqlite:///"
    if database_url.startswith(prefix):
        path = database_url.removeprefix(prefix)
    else:
        path = database_url
    return Path(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if hasattr(value, "value"):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
