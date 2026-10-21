import numpy as np
import pytest

from app.services.embeddings import l2_normalize
from app.services.vector_store import VectorStore


def unit_vectors(rows):
    return l2_normalize(np.array(rows, dtype=np.float32))


def test_search_returns_best_match_first(tmp_path):
    store = VectorStore(tmp_path / "index")
    store.add([10, 20, 30], unit_vectors([[1, 0, 0], [0, 1, 0], [1, 1, 0]]))
    hits = store.search(unit_vectors([[1, 0, 0]])[0], k=2)
    assert [chunk_id for chunk_id, _ in hits] == [10, 30]
    assert hits[0][1] == pytest.approx(1.0)


def test_search_on_empty_store(tmp_path):
    assert VectorStore(tmp_path / "index").search(np.ones(3, dtype=np.float32), k=3) == []


def test_k_larger_than_size_is_capped(tmp_path):
    store = VectorStore(tmp_path / "index")
    store.add([1], unit_vectors([[1, 0]]))
    assert len(store.search(unit_vectors([[1, 0]])[0], k=10)) == 1


def test_remove_deletes_vectors(tmp_path):
    store = VectorStore(tmp_path / "index")
    store.add([1, 2], unit_vectors([[1, 0], [0, 1]]))
    store.remove([1])
    assert store.size == 1
    assert [chunk_id for chunk_id, _ in store.search(unit_vectors([[1, 0]])[0], k=2)] == [2]


def test_save_and_reload(tmp_path):
    path = tmp_path / "nested" / "index"
    store = VectorStore(path)
    store.add([5, 6], unit_vectors([[1, 0], [0, 1]]))
    store.save()
    reloaded = VectorStore(path)
    assert reloaded.size == 2
    assert reloaded.search(unit_vectors([[0, 1]])[0], k=1)[0][0] == 6


def test_dimension_mismatch_is_rejected(tmp_path):
    store = VectorStore(tmp_path / "index")
    store.add([1], unit_vectors([[1, 0]]))
    with pytest.raises(ValueError):
        store.add([2], unit_vectors([[1, 0, 0]]))


def test_ids_and_rows_must_match(tmp_path):
    store = VectorStore(tmp_path / "index")
    with pytest.raises(ValueError):
        store.add([1, 2], unit_vectors([[1, 0]]))
