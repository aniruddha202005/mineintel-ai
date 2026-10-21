"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.deps import get_vector_store
from app.core.config import get_settings
from app.core.exceptions import MineIntelError
from app.core.logging import configure_logging
from app.db.base import Base
from app.db.session import get_engine, get_session_factory
from app.routers import documents, health, query
from app.services.index_sync import sync_vector_store

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(get_engine())
    with get_session_factory()() as db:
        sync_vector_store(db, get_vector_store())
    logger.info("%s started in %s mode", settings.app_name, settings.environment)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(MineIntelError)
    async def handle_domain_error(_: Request, exc: MineIntelError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})

    app.include_router(health.router, prefix=settings.api_prefix)
    app.include_router(documents.router, prefix=settings.api_prefix)
    app.include_router(query.router, prefix=settings.api_prefix)
    return app


app = create_app()
