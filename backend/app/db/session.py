"""Engine and session management."""

from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def build_engine(url: str) -> Engine:
    """Create an engine with options suited to the target database."""
    if url.startswith("sqlite"):
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


@lru_cache
def get_engine() -> Engine:
    return build_engine(get_settings().database_url)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)


def get_db() -> Iterator[Session]:
    """Yield a session for one request and always close it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
