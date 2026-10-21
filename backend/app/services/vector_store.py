"""Thin wrapper around a FAISS index that maps vectors to chunk ids."""

import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """Exact inner-product search over normalised vectors, keyed by chunk id.

    The index is an ``IndexFlatIP`` wrapped in ``IndexIDMap`` so chunk ids from
    the database are stored directly and individual documents can be removed.
    """

    def __init__(self, index_path: Path) -> None:
        self.index_path = Path(index_path)
        self._index: faiss.IndexIDMap | None = None
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
            except Exception:
                logger.exception("Could not read %s; starting with an empty index", self.index_path)
                self._index = None

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def add(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(ids):
            raise ValueError("vectors must be a 2D array with one row per id")
        with self._lock:
            if self._index is None:
                self._index = faiss.IndexIDMap(faiss.IndexFlatIP(matrix.shape[1]))
            elif matrix.shape[1] != self._index.d:
                raise ValueError(f"Expected vectors of dimension {self._index.d}, got {matrix.shape[1]}")
            self._index.add_with_ids(matrix, np.asarray(ids, dtype=np.int64))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to ``k`` (chunk id, similarity) pairs, best first."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            vector = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
            if vector.shape[1] != self._index.d:
                raise ValueError(f"Expected a query of dimension {self._index.d}, got {vector.shape[1]}")
            scores, ids = self._index.search(vector, min(k, self._index.ntotal))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def remove(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        with self._lock:
            if self._index is not None:
                self._index.remove_ids(np.asarray(ids, dtype=np.int64))

    def reset(self) -> None:
        with self._lock:
            self._index = None

    def save(self) -> None:
        """Write the index to disk atomically, or delete the file when the index is empty."""
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                self.index_path.unlink(missing_ok=True)
                return
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.index_path.with_suffix(".tmp")
            faiss.write_index(self._index, str(temp_path))
            os.replace(temp_path, self.index_path)
