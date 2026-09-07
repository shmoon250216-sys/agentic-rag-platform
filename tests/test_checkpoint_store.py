import pytest

from app.memory.checkpoint_store import SQLiteGraphCheckpointStore


@pytest.mark.asyncio
async def test_checkpoint_store_persists_state(tmp_path) -> None:
    store = SQLiteGraphCheckpointStore(f"sqlite:///{tmp_path / 'checkpoints.db'}")

    checkpoint_id = await store.append(
        session_id="session-1",
        route="rag",
        state={"message": "项目架构", "answer": "回答", "sources": [{"title": "文档"}]},
    )

    summaries = await store.list_for_session("session-1")
    detail = await store.get(checkpoint_id)

    assert summaries[0].checkpoint_id == checkpoint_id
    assert summaries[0].route == "rag"
    assert detail is not None
    assert detail.state["message"] == "项目架构"
    assert detail.state["sources"][0]["title"] == "文档"
