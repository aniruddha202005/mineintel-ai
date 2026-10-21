from app.db.models import Chunk, Document, QueryLog
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.repositories.query_log_repository import QueryLogRepository


def make_document(sha: str = "a" * 64, name: str = "report.pdf") -> Document:
    return Document(filename=name, stored_path=f"/tmp/{sha}.pdf", sha256=sha, page_count=1, chunk_count=2)


def test_document_lookup_by_hash(db_session):
    repo = DocumentRepository(db_session)
    document = repo.add(make_document())
    db_session.commit()
    assert repo.get_by_hash("a" * 64).id == document.id
    assert repo.get_by_hash("b" * 64) is None


def test_chunks_are_loaded_with_their_document(db_session):
    documents = DocumentRepository(db_session)
    chunks = ChunkRepository(db_session)
    document = documents.add(make_document(name="north-mine.pdf"))
    rows = chunks.add_many(
        [
            Chunk(document_id=document.id, page_number=1, chunk_index=0, text="first", embedding=b"\x00" * 8),
            Chunk(document_id=document.id, page_number=2, chunk_index=1, text="second", embedding=None),
        ]
    )
    db_session.commit()

    loaded = chunks.get_by_ids([row.id for row in rows])
    assert loaded[rows[0].id].document.filename == "north-mine.pdf"
    assert chunks.ids_for_document(document.id) == [row.id for row in rows]
    assert chunks.all_embeddings() == [(rows[0].id, b"\x00" * 8)]
    assert chunks.get_by_ids([]) == {}


def test_deleting_a_document_removes_its_chunks(db_session):
    documents = DocumentRepository(db_session)
    chunks = ChunkRepository(db_session)
    document = documents.add(make_document())
    chunks.add_many([Chunk(document_id=document.id, page_number=1, chunk_index=0, text="x" * 30)])
    db_session.commit()

    documents.delete(document)
    db_session.commit()
    assert chunks.ids_for_document(document.id) == []
    assert documents.list_all() == []


def test_recent_queries_are_newest_first_and_limited(db_session):
    repo = QueryLogRepository(db_session)
    for number in range(5):
        repo.add(QueryLog(question=f"question {number}", answer="answer", sources=[]))
    db_session.commit()
    recent = repo.recent(limit=3)
    assert [log.question for log in recent] == ["question 4", "question 3", "question 2"]
