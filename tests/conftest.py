import os
import tempfile
from pathlib import Path


# Test collection imports application singletons, so deterministic providers must
# be selected before test modules import FastAPI routes or AgentWorkflow.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["LLM_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RAG_BACKEND"] = "memory"
os.environ["RERANK_PROVIDER"] = "none"
# Singletons are constructed during collection. Never reuse the developer's DB.
_test_data = tempfile.TemporaryDirectory(prefix="agentic-rag-tests-")
os.environ["SQLITE_URL"] = f"sqlite:///{Path(_test_data.name) / 'app.db'}"
