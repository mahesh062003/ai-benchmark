"""Faithfulness and hallucination scoring, with stubbed models.

No Ollama server and no cross-encoder download: the judge is a scripted stub
and the NLI model is replaced by fixed probabilities. These tests pin the
scoring *logic* -- label assignment, NULL discipline, verdict parsing -- not
the quality of any real model's judgement.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from evaluation._shared import extract_json, split_sentences
from evaluation.faithfulness import FaithfulnessScorer
from evaluation.hallucination import HallucinationScorer
from core.settings import GenerationConfig

CONTRADICTION, ENTAILMENT, NEUTRAL = 0, 1, 2


class FakeNLI:
    """Stands in for the cross-encoder, returning scripted logits per pair."""

    def __init__(self, per_pair):
        self.per_pair = per_pair
        self.config = type("C", (), {"id2label": {0: "contradiction", 1: "entailment", 2: "neutral"}})()
        self.seen = []

    def predict(self, pairs, **kwargs):
        self.seen.extend(pairs)
        return np.array([self.per_pair[i] for i in range(len(pairs))], dtype="float64")


def scorer_with(logits, **kwargs) -> HallucinationScorer:
    scorer = HallucinationScorer(**kwargs)
    scorer._model = FakeNLI(logits)
    scorer._label_index = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return scorer


class TestSentenceSplitting:
    def test_splits_on_sentence_boundaries(self):
        assert len(split_sentences("Aspirin blocks COX-1. Clopidogrel blocks P2Y12.")) == 2

    def test_leading_option_letter_is_dropped(self):
        # "D\n\nThe patient..." must not glue the bare letter onto the first
        # sentence, which would pollute the NLI hypothesis.
        sentences = split_sentences("D\n\nThe patient has gout caused by urate crystals.")
        assert sentences == ["The patient has gout caused by urate crystals."]

    def test_short_fragments_are_dropped(self):
        assert split_sentences("Yes.") == []

    def test_empty_answer_yields_nothing(self):
        assert split_sentences("") == []


class TestHallucinationLabelling:
    def test_entailed_sentence_is_supported(self):
        scorer = scorer_with({0: [0.01, 5.0, 0.2]})
        result = scorer.score("Aspirin irreversibly blocks COX-1 in platelets.", ["premise"])
        assert result.score == 0.0
        assert result.sentences[0]["label"] == "entailment"

    def test_flatly_neutral_pair_is_neutral_not_contradiction(self):
        # Regression: entailment 0.001 vs contradiction 0.003 is a neutral
        # verdict. Comparing the two maxima alone called this a contradiction
        # and grossly overstated the contradiction rate.
        scorer = scorer_with({0: [0.1, 0.0, 6.0]})
        result = scorer.score("Some claim the context is silent about entirely.", ["premise"])
        assert result.sentences[0]["label"] == "neutral"
        assert result.contradiction_rate == 0.0
        assert result.score == 1.0  # unsupported still counts as hallucinated

    def test_confident_contradiction_is_labelled(self):
        scorer = scorer_with({0: [6.0, 0.0, 0.1]})
        result = scorer.score("A claim the context directly denies.", ["premise"])
        assert result.sentences[0]["label"] == "contradiction"
        assert result.contradiction_rate == 1.0

    def test_entailment_by_any_chunk_is_enough(self):
        # Sentence vs two chunks: unsupported by the first, entailed by the second.
        scorer = scorer_with({0: [0.1, 0.0, 6.0], 1: [0.01, 5.0, 0.2]})
        result = scorer.score("A claim supported by the second chunk only.", ["a", "b"])
        assert result.score == 0.0

    def test_score_is_the_unsupported_fraction(self):
        scorer = scorer_with({0: [0.01, 5.0, 0.2], 1: [0.1, 0.0, 6.0]})
        result = scorer.score(
            "This first claim is fully supported by context. "
            "This second claim is not mentioned anywhere at all.",
            ["premise"],
        )
        assert result.score == pytest.approx(0.5)

    def test_detail_json_round_trips(self):
        scorer = scorer_with({0: [0.01, 5.0, 0.2]})
        detail = json.loads(scorer.score("A supported claim about the context.", ["p"]).to_json())
        assert detail["n_sentences"] == 1
        assert detail["nli_model"]


class TestHallucinationNulls:
    def test_empty_answer_is_null_not_zero(self):
        assert scorer_with({}).score("", ["premise"]).score is None

    def test_missing_context_is_null_not_one(self):
        # No context retrieved is an unmeasurable case, not total hallucination.
        assert scorer_with({}).score("Some answer sentence here.", []).score is None

    def test_answer_without_scoreable_sentences_is_null(self):
        assert scorer_with({}).score("Yes.", ["premise"]).score is None

    def test_batch_scoring_matches_single_scoring(self):
        logits = {0: [0.01, 5.0, 0.2], 1: [0.1, 0.0, 6.0]}
        single = scorer_with(logits).score("A supported claim about the thing.", ["a", "b"])
        batched = scorer_with(logits).score_batch([("A supported claim about the thing.", ["a", "b"])])
        assert batched[0].score == single.score


class StubJudge:
    """A judge returning scripted replies, one per call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        if not self.replies:
            raise AssertionError("judge called more times than scripted")
        return self.replies.pop(0)


def faithfulness_with(*replies) -> FaithfulnessScorer:
    scorer = FaithfulnessScorer.__new__(FaithfulnessScorer)
    scorer.judge_model = "stub-judge"
    scorer.client = StubJudge(*replies)
    return scorer


class TestFaithfulness:
    def test_supported_fraction_is_the_score(self):
        scorer = faithfulness_with(
            '["Aspirin blocks COX-1.", "Aspirin is a beta blocker.", "Aspirin is oral."]',
            '[{"index":1,"verdict":1},{"index":2,"verdict":0},{"index":3,"verdict":1}]',
        )
        result = scorer.score("q", "Aspirin blocks COX-1.", ["context"])
        assert result.score == pytest.approx(2 / 3)
        assert result.judge == "stub-judge"

    def test_fully_grounded_answer_scores_one(self):
        scorer = faithfulness_with('["One claim."]', '[{"index":1,"verdict":1}]')
        assert scorer.score("q", "One claim.", ["context"]).score == 1.0

    def test_json_inside_prose_and_fences_is_recovered(self):
        scorer = faithfulness_with(
            'Sure! Here you go:\n```json\n["A claim."]\n```',
            'Verdicts:\n```json\n[{"index":1,"verdict":"yes"}]\n```',
        )
        assert scorer.score("q", "A claim.", ["context"]).score == 1.0

    def test_unreadable_verdict_counts_as_unsupported(self):
        # Conservative direction: it can understate faithfulness, never invent support.
        scorer = faithfulness_with('["A claim."]', '[{"index":1,"verdict":"???"}]')
        assert scorer.score("q", "A claim.", ["context"]).score == 0.0

    def test_answer_with_no_statements_is_null_not_zero(self):
        scorer = faithfulness_with("[]")
        result = scorer.score("q", "I cannot answer from the context.", ["context"])
        assert result.score is None
        assert "no factual statements" in result.error

    def test_unparseable_judge_output_is_null(self):
        scorer = faithfulness_with('["A claim."]', "I could not comply.")
        assert scorer.score("q", "A claim.", ["context"]).score is None

    def test_empty_answer_and_missing_context_are_null(self):
        assert faithfulness_with().score("q", "", ["c"]).score is None
        assert faithfulness_with().score("q", "An answer.", []).score is None

    def test_judge_failure_is_null_not_zero(self):
        class Boom:
            def generate(self, prompt):
                raise ConnectionError("ollama unreachable")

        scorer = faithfulness_with()
        scorer.client = Boom()
        result = scorer.score("q", "An answer.", ["context"])
        assert result.score is None
        assert "unreachable" in result.error

    def test_context_reaches_the_verification_prompt(self):
        scorer = faithfulness_with('["A claim."]', '[{"index":1,"verdict":1}]')
        scorer.score("q", "A claim.", ["a distinctive premise string"])
        assert "a distinctive premise string" in scorer.client.prompts[1]


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('["a"]') == ["a"]

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_json_embedded_in_prose(self):
        assert extract_json('Here it is: [1, 2] hope that helps') == [1, 2]

    def test_unparseable_returns_none(self):
        assert extract_json("no json at all") is None
        assert extract_json("") is None


class TestStratifiedSample:
    """Faithfulness is judged on a subsample; every comparison cell must survive it."""

    @staticmethod
    def population(datasets=2, methods=3, models=4, per_cell=50):
        return [
            {"dataset": f"d{d}", "method": f"m{m}", "model": f"L{n}", "query_id": f"q{q}"}
            for d in range(datasets)
            for m in range(methods)
            for n in range(models)
            for q in range(per_cell)
        ]

    def test_every_cell_is_represented(self):
        from generation.runner import stratified_sample

        rows = self.population()
        picked = stratified_sample(rows, 240, seed=42)
        cells = {(r["dataset"], r["method"], r["model"]) for r in picked}
        assert len(picked) == 240
        assert len(cells) == 2 * 3 * 4  # no cell dropped

    def test_allocation_is_even_across_cells(self):
        from collections import Counter

        from generation.runner import stratified_sample

        picked = stratified_sample(self.population(), 240, seed=42)
        counts = Counter((r["dataset"], r["method"], r["model"]) for r in picked)
        assert max(counts.values()) - min(counts.values()) <= 1

    def test_small_cells_do_not_starve_large_ones(self):
        from generation.runner import stratified_sample

        rows = [{"dataset": "big", "method": "m", "model": "L", "query_id": str(i)}
                for i in range(100)]
        rows += [{"dataset": "tiny", "method": "m", "model": "L", "query_id": "t0"}]
        picked = stratified_sample(rows, 50, seed=1)
        # The one-row cell contributes its single row; the rest come from "big".
        assert len(picked) == 50
        assert sum(r["dataset"] == "tiny" for r in picked) == 1

    def test_sample_is_deterministic_for_a_seed(self):
        from generation.runner import stratified_sample

        rows = self.population()
        first = stratified_sample(rows, 120, seed=7)
        second = stratified_sample(rows, 120, seed=7)
        assert [r["query_id"] for r in first] == [r["query_id"] for r in second]

    def test_requesting_more_than_available_returns_everything(self):
        from generation.runner import stratified_sample

        rows = self.population(per_cell=2)
        assert len(stratified_sample(rows, 10_000, seed=1)) == len(rows)

    def test_no_duplicates_are_drawn(self):
        from generation.runner import stratified_sample

        picked = stratified_sample(self.population(), 300, seed=3)
        keys = [(r["dataset"], r["method"], r["model"], r["query_id"]) for r in picked]
        assert len(keys) == len(set(keys))


def test_scoring_defaults_are_recorded_in_config():
    from core.settings import Config

    scoring = Config().scoring
    assert scoring.nli_model
    assert 0.0 < scoring.entail_threshold <= 1.0
    assert 0.0 < scoring.contradict_threshold <= 1.0


def test_generation_config_lists_the_compared_models():
    assert GenerationConfig().models == ["llama3.1", "gemma2", "mistral", "qwen2.5"]
