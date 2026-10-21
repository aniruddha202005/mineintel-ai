"""Runs a question and records it in the query history."""

import time

from sqlalchemy.orm import Session

from app.db.models import QueryLog
from app.repositories.query_log_repository import QueryLogRepository
from app.services.qa_service import AnswerResult, QAService


class QueryService:
    def __init__(self, db: Session, qa: QAService) -> None:
        self.db = db
        self.qa = qa
        self.logs = QueryLogRepository(db)

    def run(self, question: str, top_k: int | None, agentic: bool) -> tuple[QueryLog, AnswerResult]:
        started = time.perf_counter()
        result = self.qa.ask(question, top_k=top_k, agentic=agentic)
        latency_ms = int((time.perf_counter() - started) * 1000)

        log = self.logs.add(
            QueryLog(
                question=question,
                answer=result.answer,
                grounded=result.grounded,
                agentic=agentic,
                rewritten_query=result.rewritten_query,
                sources=[
                    {
                        "index": source.index,
                        "document_id": source.chunk.document_id,
                        "document_name": source.chunk.document_name,
                        "page_number": source.chunk.page_number,
                        "chunk_id": source.chunk.chunk_id,
                        "score": round(source.chunk.score, 4),
                    }
                    for source in result.sources
                ],
                latency_ms=latency_ms,
            )
        )
        self.db.commit()
        return log, result

    def history(self, limit: int = 20) -> list[QueryLog]:
        return self.logs.recent(limit)
