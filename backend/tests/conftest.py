import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.api import deps


@pytest.fixture
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def isolated_app_state(tmp_path, monkeypatch):
    """Point the application at a temporary database and data directory."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OCR_ENABLED", "false")
    monkeypatch.setenv("MIN_RELEVANCE_SCORE", "0.2")
    caches = [get_settings, get_engine, get_session_factory, deps.get_vector_store, deps.get_embedder, deps.get_llm]
    for cache in caches:
        cache.cache_clear()
    yield tmp_path
    for cache in caches:
        cache.cache_clear()
