"""Question answering and history."""

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_query_service
from app.schemas.query import QueryHistoryItem, QueryRequest, QueryResponse, SourceOut
from app.services.query_service import QueryService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def ask_question(payload: QueryRequest, service: QueryService = Depends(get_query_service)) -> QueryResponse:
    log, result = service.run(payload.question, payload.top_k, payload.agentic)
    return QueryResponse(
        id=log.id,
        question=log.question,
        answer=result.answer,
        grounded=result.grounded,
        agentic=payload.agentic,
        rewritten_query=result.rewritten_query,
        latency_ms=log.latency_ms,
        sources=[
            SourceOut(
                index=source.index,
                document_id=source.chunk.document_id,
                document_name=source.chunk.document_name,
                page_number=source.chunk.page_number,
                chunk_id=source.chunk.chunk_id,
                score=round(source.chunk.score, 4),
                text=source.chunk.text,
            )
            for source in result.sources
        ],
    )


@router.get("/history", response_model=list[QueryHistoryItem])
def query_history(
    limit: int = Query(default=20, ge=1, le=100),
    service: QueryService = Depends(get_query_service),
) -> list[QueryHistoryItem]:
    return service.history(limit)
