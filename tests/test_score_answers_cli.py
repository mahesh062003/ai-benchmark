"""The score-answers pass: sampling, NULL discipline, and rater recording.

The CLI scoring helpers take their scorer and chunk-text loader as arguments,
so these tests drive the real database code with stubs in place of the LLM
judge and the cross-encoder. No Ollama server and no model download.
"""

from __future__ import annotations

import pytest

from cli import _score_faithfulness, _score_hallucination
from core.settings import Config, PathConfig
from core.db import Database

DATASETS = ("cuad", "medqa")
METHODS = ("bm25", "dense", "hybrid")
MODELS = ("llama3.1", "gemma2", "mistral", "qwen2.5")


@pytest.fixture
def config(tmp_path) -> Config:
    cfg = Config()
    cfg.paths = PathConfig(
        datasets=str(tmp_path / "datasets"), artifacts=str(tmp_path / "artifacts"),
        corpora=str(tmp_path / "corpora"), indexes=str(tmp_path / "indexes"),
        results=str(tmp_path / "results"), database=str(tmp_path / "b.sqlite"),
    )
    return cfg


@pytest.fixture
def populated(tmp_path, config) -> Database:
    """A run with an answer in every (dataset, method, model) cell."""
    database = Database(tmp_path / "b.sqlite")
    database.create_run("run-1", stage="generation-multi-model", config=config)
    for dataset in DATASETS:
        for method in METHODS:
            for model in MODELS:
                for q in range(10):
                    database.save_generation(
                        "run-1", dataset, method, model,
                        {
                            "query_id": f"q{q}", "prompt": "Question: why?\n",
                            "answer": "A grounded claim about the retrieved context.",
                            "reference_answer": "x", "context_chunk_ids": ["c1"],
                            "latency_ms": 10.0, "exact_match": None,
                            "choice_correct": None,
                        },
                    )
    yield database
    database.close()


def fake_chunk_texts(cfg, dataset, needed):
    return {chunk_id: "a premise sentence" for chunk_id in needed}


class StubFaithfulness:
    """Records how many rows it was asked to judge."""

    instances = []

    def __init__(self, generation_config, judge_model="stub"):
        self.judge_model = judge_model
        self.calls = 0
        StubFaithfulness.instances.append(self)

    def available(self):
        return True

    def score(self, question, answer, context):
        self.calls += 1
        return type(
            "R", (), {"score": 0.5, "judge": self.judge_model, "to_json": lambda s: "{}"}
        )()


class StubHallucination:
    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "stub-nli")

    def score_batch(self, items):
        return [
            type("R", (), {"score": 0.25, "model": self.model_name,
                           "to_json": lambda s: "{}"})()
            for _ in items
        ]


class TestFaithfulnessSampling:
    def test_sample_limits_rows_and_leaves_the_rest_null(self, populated, config):
        StubFaithfulness.instances.clear()
        config.scoring.faithfulness_sample = 48
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)

        scored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE faithfulness IS NOT NULL"
        ).fetchone()[0]
        unscored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE faithfulness IS NULL"
        ).fetchone()[0]
        assert scored == 48
        assert unscored == 240 - 48          # the rest stay NULL, not 0.0
        assert StubFaithfulness.instances[0].calls == 48

    def test_sample_covers_every_cell(self, populated, config):
        config.scoring.faithfulness_sample = 48   # 2 datasets x 3 methods x 4 models
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        cells = populated.connection.execute(
            "SELECT count(*) FROM (SELECT DISTINCT dataset, method, model FROM generations "
            "WHERE faithfulness IS NOT NULL)"
        ).fetchone()[0]
        assert cells == len(DATASETS) * len(METHODS) * len(MODELS)

    def test_every_model_is_judged_on_the_same_questions(self, populated, config):
        """What makes the model comparison paired rather than unpaired."""
        config.scoring.faithfulness_sample = 48
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)

        rows = populated.connection.execute(
            "SELECT dataset, method, model, query_id FROM generations "
            "WHERE faithfulness IS NOT NULL"
        ).fetchall()
        by_model = {}
        for dataset, method, model, query_id in rows:
            by_model.setdefault(model, set()).add((dataset, method, query_id))

        assert len(by_model) == len(MODELS)
        reference = next(iter(by_model.values()))
        assert all(questions == reference for questions in by_model.values())

    def test_no_sample_scores_everything(self, populated, config):
        config.scoring.faithfulness_sample = None
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        scored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE faithfulness IS NOT NULL"
        ).fetchone()[0]
        assert scored == 240

    def test_judge_is_recorded_on_every_scored_row(self, populated, config):
        config.scoring.faithfulness_sample = 24
        config.scoring.judge_model = "mistral"
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        judges = {
            row[0] for row in populated.connection.execute(
                "SELECT DISTINCT faithfulness_judge FROM generations "
                "WHERE faithfulness IS NOT NULL"
            )
        }
        assert judges == {"mistral"}

    def test_rerun_only_scores_what_is_still_null(self, populated, config):
        StubFaithfulness.instances.clear()
        config.scoring.faithfulness_sample = 24
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        assert [i.calls for i in StubFaithfulness.instances] == [24, 24]
        scored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE faithfulness IS NOT NULL"
        ).fetchone()[0]
        assert scored == 48   # second pass added 24 more, did not redo the first

    def test_unavailable_judge_scores_nothing(self, populated, config):
        class Unavailable(StubFaithfulness):
            def available(self):
                return False

        _score_faithfulness(config, populated, None, None, fake_chunk_texts, Unavailable)
        scored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE faithfulness IS NOT NULL"
        ).fetchone()[0]
        assert scored == 0


class TestHallucinationCoverage:
    def test_hallucination_covers_every_answer(self, populated, config):
        # Cheap on GPU, so it is not subsampled the way faithfulness is.
        _score_hallucination(config, populated, None, None, fake_chunk_texts, StubHallucination)
        scored = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE hallucination IS NOT NULL"
        ).fetchone()[0]
        assert scored == 240

    def test_nli_model_is_recorded(self, populated, config):
        _score_hallucination(config, populated, None, None, fake_chunk_texts, StubHallucination)
        raters = {
            row[0] for row in populated.connection.execute(
                "SELECT DISTINCT hallucination_model FROM generations "
                "WHERE hallucination IS NOT NULL"
            )
        }
        assert raters == {config.scoring.nli_model}

    def test_the_two_measures_are_independent(self, populated, config):
        config.scoring.faithfulness_sample = 24
        _score_hallucination(config, populated, None, None, fake_chunk_texts, StubHallucination)
        _score_faithfulness(config, populated, None, None, fake_chunk_texts, StubFaithfulness)
        row = populated.connection.execute(
            "SELECT count(*) FROM generations WHERE hallucination IS NOT NULL "
            "AND faithfulness IS NULL"
        ).fetchone()[0]
        # Most answers have a hallucination score but no faithfulness score,
        # which is exactly the intended asymmetry.
        assert row == 240 - 24
