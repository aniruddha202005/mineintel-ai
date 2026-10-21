"""Keep the FAISS index consistent with the embeddings stored in the database."""

import logging

import numpy as np
from sqlalchemy.orm import Session

from app.repositories.chunk_repository import ChunkRepository
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def sync_vector_store(db: Session, store: VectorStore) -> bool:
    """Rebuild the index from the database if the two disagree.

    Hosting platforms with an ephemeral disk lose the index file on restart.
    Because every embedding is also stored in the database, the index can be
    rebuilt without calling the embedding model again.

    Returns True when a rebuild happened.
    """
    rows = ChunkRepository(db).all_embeddings()
    if store.size == len(rows):
        return False

    logger.info("Rebuilding vector index: %s in index, %s in database", store.size, len(rows))
    store.reset()
    if rows:
        ids = [chunk_id for chunk_id, _ in rows]
        matrix = np.vstack([np.frombuffer(blob, dtype=np.float32) for _, blob in rows])
        store.add(ids, matrix)
    store.save()
    return True
