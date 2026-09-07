import sqlite3

import pytest

from app.memory.sqlite_store import SQLiteSessionStore


@pytest.mark.asyncio
async def test_sqlite_store_persists_messages_across_instances(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'memory.db'}"
    first_store = SQLiteSessionStore(database_url)

    session = await first_store.get_or_create("user-1")
    await first_store.append(session.session_id, "user", "你好，今天适合学习 Agent")
    await first_store.append(session.session_id, "assistant", "你好，我是 Agent")

    second_store = SQLiteSessionStore(database_url)
    sessions = await second_store.list_sessions(user_id="user-1")
    messages = await second_store.list_messages(session.session_id)

    assert sessions[0].session_id == session.session_id
    assert sessions[0].title == "今天适合学习 Agent"
    assert sessions[0].message_count == 2
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[1].content == "你好，我是 Agent"


@pytest.mark.asyncio
async def test_sqlite_store_keeps_title_from_first_user_message(tmp_path) -> None:
    store = SQLiteSessionStore(f"sqlite:///{tmp_path / 'titles.db'}")
    session = await store.get_or_create("user-1")

    await store.append(session.session_id, "user", "请帮我分析 Agent 路由机制")
    await store.append(session.session_id, "assistant", "开始分析")
    await store.append(session.session_id, "user", "再解释一下 RAG")

    sessions = await store.list_sessions(user_id="user-1")

    assert sessions[0].title == "分析 Agent 路由机制"


@pytest.mark.asyncio
async def test_sqlite_store_backfills_titles_for_legacy_database(tmp_path) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as db:
        db.execute(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute(
            """
            CREATE TABLE messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        db.execute("INSERT INTO sessions (session_id, user_id) VALUES ('legacy-1', 'user-1')")
        db.execute(
            """
            INSERT INTO messages (session_id, role, content)
            VALUES ('legacy-1', 'user', '请解释长期记忆的工作原理')
            """
        )

    store = SQLiteSessionStore(f"sqlite:///{database_path}")
    sessions = await store.list_sessions(user_id="user-1")

    assert sessions[0].title == "解释长期记忆的工作原理"
