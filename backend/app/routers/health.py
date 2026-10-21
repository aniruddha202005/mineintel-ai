"""Health check."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_vector_store
from app.db.session import get_db
from app.services.vector_store import VectorStore

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db), store: VectorStore = Depends(get_vector_store)) -> dict:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "indexed_chunks": store.size}
