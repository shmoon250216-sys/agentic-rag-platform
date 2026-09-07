from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    rag_index_name: str = "idx:rag_chunks"
    rag_vector_dimensions: int = 256
    redis_url: str = "redis://localhost:6379/0"
    redis_connect_timeout_seconds: float = 2
    redis_socket_timeout_seconds: float = 5
    sqlite_url: str = "sqlite:///./data/app.db"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


@lru_cache
def get_settings() -> Settings:
    return Settings()
