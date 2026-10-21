"""Query request and response schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    agentic: bool = False


class SourceOut(BaseModel):
    index: int
    document_id: int
    document_name: str
    page_number: int
    chunk_id: int
    score: float
    text: str


class QueryResponse(BaseModel):
    id: int
    question: str
    answer: str
    grounded: bool
    agentic: bool
    rewritten_query: str | None
    sources: list[SourceOut]
    latency_ms: int


class QueryHistoryItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    question: str
    answer: str
    grounded: bool
    agentic: bool
    rewritten_query: str | None
    latency_ms: int
    created_at: datetime
