"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the whole application."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "MineIntel AI"
    environment: str = "development"
    log_level: str = "INFO"
    api_prefix: str = "/api"
    cors_origins: str = "http://localhost:5173"

    database_url: str = "sqlite:///./mineintel.db"
    data_dir: Path = Path("./data")
    max_upload_mb: int = 25

    chunk_size: int = 1000
    chunk_overlap: int = 150
    min_page_chars: int = 40
    ocr_enabled: bool = True
    ocr_dpi: int = 200
    ocr_language: str = "eng"

    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_api_token: str = ""

    top_k: int = 5
    min_relevance_score: float = 0.25
    max_query_rewrites: int = 2

    llm_provider: str = "openai_compatible"
    llm_model: str = "llama-3.1-8b-instant"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 700

    @field_validator("database_url")
    @classmethod
    def normalize_database_url(cls, value: str) -> str:
        """Make hosted Postgres URLs compatible with the psycopg2 driver."""
        if value.startswith("postgres://"):
            return value.replace("postgres://", "postgresql+psycopg2://", 1)
        if value.startswith("postgresql://"):
            return value.replace("postgresql://", "postgresql+psycopg2://", 1)
        return value

    @model_validator(mode="after")
    def check_chunk_settings(self) -> "Settings":
        """Reject chunk settings that would never terminate or lose text."""
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be at least 0 and smaller than CHUNK_SIZE")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "faiss.index"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
