"""Generation stage with a stubbed Ollama client.

No live Ollama server is required. These tests cover prompt assembly, the
answer-quality measures, and error handling; they do NOT verify the quality of
any real model's output.
"""

from __future__ import annotations

import pytest

from core.settings import Config
from generation.generator import GenerationEvaluation, generate_for_query
from core.types import Chunk, Query


class StubClient:
    def __init__(self, response="B. because the context says so", fail=False):
        self.response = response
        self.fail = fail
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if self.fail:
            raise ConnectionError("ollama unreachable")
        return self.response


@pytest.fixture
def chunks():
    return [
        Chunk(f"c{i}", "d", "ds", "s", f"context passage {i}", i, 0, 10) for i in range(3)
    ]


@pytest.fixture
def mcq_query():
    return Query(
        "q1", "medqa", "test", "Which drug is indicated?",
        answer="Paracetamol",
        metadata={
            "options": {"A": "Aspirin", "B": "Paracetamol", "C": "Ibuprofen"},
            "answer_idx": "B",
            "answer_type": "multiple_choice",
        },
    )


class TestGenerateForQuery:
    def test_correct_choice_scored_one(self, mcq_query, chunks):
        result = generate_for_query(StubClient("B"), mcq_query, chunks, Config())
        assert result.choice_correct == 1.0
        assert result.error is None
        assert result.latency_ms is not None

    def test_wrong_choice_scored_zero(self, mcq_query, chunks):
        result = generate_for_query(StubClient("A"), mcq_query, chunks, Config())
        assert result.choice_correct == 0.0

    def test_context_ids_recorded_for_traceability(self, mcq_query, chunks):
        config = Config()
        config.generation.max_context_chunks = 2
        result = generate_for_query(StubClient(), mcq_query, chunks, config)
        assert result.context_chunk_ids == ["c0", "c1"]

    def test_prompt_contains_retrieved_context(self, mcq_query, chunks):
        client = StubClient()
        generate_for_query(client, mcq_query, chunks, Config())
        assert "context passage 0" in client.prompts[0]

    def test_grounding_metrics_are_null_until_scored_separately(self, mcq_query, chunks):
        # Faithfulness and hallucination are a second pass (`score-answers`);
        # the generation loop must never write a value for them.
        result = generate_for_query(StubClient("B"), mcq_query, chunks, Config())
        assert result.faithfulness is None
        assert result.hallucination is None

    def test_failure_is_recorded_not_swallowed(self, mcq_query, chunks):
        result = generate_for_query(StubClient(fail=True), mcq_query, chunks, Config())
        assert result.answer is None
        assert "unreachable" in result.error
        # A failed generation must not be scored as a wrong answer.
        assert result.choice_correct is None
        assert result.exact_match is None

    def test_non_mcq_dataset_gets_null_choice_score(self, chunks):
        query = Query("q", "pubmedqa", "labeled", "Does X cause Y?", answer="yes")
        result = generate_for_query(StubClient("yes"), query, chunks, Config())
        assert result.choice_correct is None      # no options -> not applicable
        assert result.exact_match == 1.0          # but the answer is checkable

    def test_record_serialises_for_database(self, mcq_query, chunks):
        record = generate_for_query(StubClient("B"), mcq_query, chunks, Config()).to_record()
        assert record["query_id"] == "q1"
        assert record["faithfulness"] is None
        assert set(record) >= {
            "query_id", "prompt", "answer", "reference_answer", "context_chunk_ids",
            "latency_ms", "exact_match", "choice_correct", "faithfulness",
            "hallucination", "error",
        }


def test_generation_disabled_by_default():
    assert Config().generation.enabled is False


def test_evaluation_dataclass_defaults_to_null_quality_metrics():
    evaluation = GenerationEvaluation(
        query_id="q", prompt="p", answer="a", reference_answer="r",
        context_chunk_ids=[], latency_ms=1.0, exact_match=None, choice_correct=None,
    )
    assert evaluation.faithfulness is None
    assert evaluation.hallucination is None
