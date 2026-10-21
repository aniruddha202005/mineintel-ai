"""Grounded question answering."""

import logging
import re
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.agent_graph import build_agent_graph
from app.services.prompts import ANSWER_PROMPT, INSUFFICIENT_CONTEXT, REWRITE_PROMPT
from app.services.retrieval_service import RetrievalService, RetrievedChunk

logger = logging.getLogger(__name__)

NOT_FOUND_MESSAGE = "The uploaded documents do not contain enough information to answer this question."
_CITATION = re.compile(r"\[(\d+)\]")


@dataclass
class Source:
    """A passage shown to the model, numbered the way it is cited in the answer."""

    index: int
    chunk: RetrievedChunk


@dataclass
class AnswerResult:
    answer: str
    grounded: bool
    sources: list[Source] = field(default_factory=list)
    rewritten_query: str | None = None


class QAService:
    """Answers questions strictly from retrieved passages."""

    def __init__(self, retrieval: RetrievalService, llm: BaseChatModel, settings: Settings) -> None:
        self.retrieval = retrieval
        self.settings = settings
        self._answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        self._rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
        self._graph = build_agent_graph(
            retrieve=self.retrieval.retrieve,
            keep_relevant=self.filter_relevant,
            rewrite=self.rewrite_query,
            generate=self.generate,
            max_rewrites=settings.max_query_rewrites,
        )

    def ask(self, question: str, top_k: int | None = None, agentic: bool = False) -> AnswerResult:
        """Answer ``question`` using the plain pipeline or the LangGraph workflow."""
        k = top_k or self.settings.top_k
        if agentic:
            return self._graph.invoke({"question": question, "query": question, "top_k": k, "attempts": 0})["result"]
        chunks = self.filter_relevant(self.retrieval.retrieve(question, k))
        return self.generate(question, chunks)

    def filter_relevant(self, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Keep only chunks whose similarity reaches the configured threshold."""
        return [chunk for chunk in chunks if chunk.score >= self.settings.min_relevance_score]

    def rewrite_query(self, question: str, query: str) -> str:
        """Ask the model for a better search query."""
        try:
            rewritten = self._rewrite_chain.invoke({"question": question, "query": query})
        except Exception as exc:
            raise LLMError(f"Query rewrite failed: {exc}") from exc
        rewritten = rewritten.strip().splitlines()[0].strip() if rewritten.strip() else ""
        return rewritten or query

    def generate(self, question: str, chunks: list[RetrievedChunk], rewritten_query: str | None = None) -> AnswerResult:
        """Produce an answer from the given chunks, or report that the documents lack one."""
        if not chunks:
            return AnswerResult(NOT_FOUND_MESSAGE, grounded=False, rewritten_query=rewritten_query)

        sources = [Source(index=position, chunk=chunk) for position, chunk in enumerate(chunks, start=1)]
        context = "\n\n".join(
            f"[{source.index}] ({source.chunk.document_name}, page {source.chunk.page_number})\n{source.chunk.text}"
            for source in sources
        )
        try:
            answer = self._answer_chain.invoke({"context": context, "question": question}).strip()
        except Exception as exc:
            raise LLMError(f"The language model request failed: {exc}") from exc

        if not answer or INSUFFICIENT_CONTEXT in answer:
            return AnswerResult(NOT_FOUND_MESSAGE, grounded=False, rewritten_query=rewritten_query)

        cited = {int(number) for number in _CITATION.findall(answer)}
        used = [source for source in sources if source.index in cited] or sources
        return AnswerResult(answer, grounded=True, sources=used, rewritten_query=rewritten_query)
