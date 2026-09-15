from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "Adaptive Agentic RAG Platform"
    app_env: str = "local"
    api_token: str = "dev-token"
    llm_provider: str = "fake"
    llm_api_key: str | None = None
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 30
    embedding_provider: str = "hash"
    embedding_api_key: str | None = None
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 256
    embedding_timeout_seconds: float = 30
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 256
    document_upload_max_bytes: int = 10 * 1024 * 1024
    document_extract_max_chars: int = 200_000
    document_pdf_max_pages: int = 300
    document_docx_max_uncompressed_bytes: int = 50 * 1024 * 1024
    quality_min_total_cases: int = 12
    quality_min_route_accuracy: float = 0.95
    quality_min_rag_hit_rate: float = 0.9
    quality_min_tool_accuracy: float = 0.95
    quality_max_average_latency_ms: float = 500
    rag_backend: str = "memory"
    rag_index_name: str = "idx:rag_chunks:v2"
    rag_vector_dimensions: int = 256
    rag_candidate_multiplier: int = 4
    rag_rrf_k: int = 60
    rag_bm25_weight: float = 1.0
    rag_vector_weight: float = 0.1
    rerank_provider: str = "none"
    rerank_url: str = ""
    rerank_api_key: str | None = None
    rerank_model: str = ""
    rerank_candidate_count: int = 12
    rerank_timeout_seconds: float = 15
    rerank_fail_open: bool = True
    redis_url: str = "redis://localhost:6379/0"
    redis_connect_timeout_seconds: float = 2
    redis_socket_timeout_seconds: float = 5
    sqlite_url: str = "sqlite:///./data/app.db"
    context_recent_messages: int = Field(default=6, ge=2, le=30)
    context_history_chars: int = Field(default=6000, ge=500, le=30000)
    context_message_chars: int = Field(default=2000, ge=100, le=4000)
    context_summary_chars: int = Field(default=1200, ge=100, le=4000)
    session_max_messages: int = Field(default=100, ge=10, le=1000)
    session_max_checkpoints: int = Field(default=30, ge=1, le=1000)
    memory_max_items: int = Field(default=100, ge=5, le=1000)
    memory_ttl_days: int = Field(default=180, ge=1, le=3650)
    memory_item_chars: int = Field(default=320, ge=80, le=1000)

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
