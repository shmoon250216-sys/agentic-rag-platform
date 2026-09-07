import os


# Test collection imports application singletons, so deterministic providers must
# be selected before test modules import FastAPI routes or AgentWorkflow.
os.environ["LLM_PROVIDER"] = "fake"
os.environ["LLM_API_KEY"] = ""
os.environ["EMBEDDING_PROVIDER"] = "hash"
os.environ["RAG_BACKEND"] = "memory"
