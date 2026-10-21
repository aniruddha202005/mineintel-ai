import pytest
from langchain_core.runnables import RunnableLambda

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.services.prompts import INSUFFICIENT_CONTEXT
from app.services.qa_service import NOT_FOUND_MESSAGE, QAService
from app.services.retrieval_service import RetrievedChunk
from tests.fakes import fake_llm


def make_chunk(chunk_id: int, text: str, score: float, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=1,
        document_name="report.pdf",
        page_number=page,
        chunk_index=chunk_id,
        text=text,
        score=score,
    )


class FakeRetrieval:
    """Returns canned chunks per query and records the queries it received."""

    def __init__(self, results_by_query: dict[str, list[RetrievedChunk]]) -> None:
        self.results_by_query = results_by_query
        self.queries: list[str] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.results_by_query.get(query, [])[:top_k]


@pytest.fixture
def settings():
    return Settings(min_relevance_score=0.3, max_query_rewrites=2, top_k=3)


def test_answer_is_grounded_and_cites_only_used_passages(settings):
    chunks = [make_chunk(1, "Loose strata caused the fall.", 0.8), make_chunk(2, "Unrelated text.", 0.5, page=4)]
    qa = QAService(FakeRetrieval({"why": chunks}), fake_llm("Loose strata caused it [1]."), settings)
    result = qa.ask("why")
    assert result.grounded is True
    assert [source.index for source in result.sources] == [1]
    assert result.sources[0].chunk.page_number == 1


def test_all_sources_returned_when_model_cites_nothing(settings):
    chunks = [make_chunk(1, "Text one.", 0.8), make_chunk(2, "Text two.", 0.7)]
    qa = QAService(FakeRetrieval({"q": chunks}), fake_llm("An answer without brackets."), settings)
    assert len(qa.ask("q").sources) == 2


def test_model_can_decline_when_context_is_insufficient(settings):
    qa = QAService(FakeRetrieval({"q": [make_chunk(1, "Text.", 0.9)]}), fake_llm(INSUFFICIENT_CONTEXT), settings)
    result = qa.ask("q")
    assert result.grounded is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


def test_llm_is_not_called_when_nothing_is_relevant(settings):
    calls = []
    llm = RunnableLambda(lambda prompt: calls.append(prompt) or "should not happen")
    qa = QAService(FakeRetrieval({"q": [make_chunk(1, "Weak match.", 0.1)]}), llm, settings)
    result = qa.ask("q")
    assert result.grounded is False
    assert result.answer == NOT_FOUND_MESSAGE
    assert calls == []


def test_top_k_is_passed_to_retrieval(settings):
    chunks = [make_chunk(number, f"Text {number}.", 0.9) for number in range(1, 6)]
    retrieval = FakeRetrieval({"q": chunks})
    qa = QAService(retrieval, fake_llm("Fine [1]."), settings)
    assert len(qa.ask("q", top_k=2).sources) == 1
    assert retrieval.queries == ["q"]


def test_agentic_flow_rewrites_weak_queries():
    settings = Settings(min_relevance_score=0.3, max_query_rewrites=2, top_k=3)
    good = [make_chunk(1, "Strata control failure.", 0.9)]
    retrieval = FakeRetrieval({"rock fall": [make_chunk(9, "Weak.", 0.1)], "roof fall strata failure": good})

    def respond(prompt):
        text = prompt.to_string()
        return "roof fall strata failure" if "Rewritten query" in text else "Strata failed [1]."

    qa = QAService(retrieval, RunnableLambda(respond), settings)
    result = qa.ask("rock fall", agentic=True)
    assert retrieval.queries == ["rock fall", "roof fall strata failure"]
    assert result.grounded is True
    assert result.rewritten_query == "roof fall strata failure"


def test_agentic_flow_stops_after_max_rewrites():
    settings = Settings(min_relevance_score=0.3, max_query_rewrites=2, top_k=3)
    retrieval = FakeRetrieval({})
    qa = QAService(retrieval, fake_llm("another attempt"), settings)
    result = qa.ask("unanswerable", agentic=True)
    assert len(retrieval.queries) == 3
    assert result.grounded is False
    assert result.answer == NOT_FOUND_MESSAGE


def test_agentic_flow_skips_rewrite_when_first_retrieval_is_good(settings):
    retrieval = FakeRetrieval({"q": [make_chunk(1, "Good match.", 0.9)]})
    qa = QAService(retrieval, fake_llm("Answer [1]."), settings)
    result = qa.ask("q", agentic=True)
    assert retrieval.queries == ["q"]
    assert result.rewritten_query is None


def test_llm_failures_are_reported_as_llm_errors(settings):
    def explode(_):
        raise RuntimeError("connection refused")

    qa = QAService(FakeRetrieval({"q": [make_chunk(1, "Text.", 0.9)]}), RunnableLambda(explode), settings)
    with pytest.raises(LLMError):
        qa.ask("q")
