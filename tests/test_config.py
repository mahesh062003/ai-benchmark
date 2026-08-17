"""Configuration, IDs and generation helpers."""

from __future__ import annotations

import re

import pytest

from core.settings import PROJECT_ROOT, ChunkConfig, Config, load_config
from generation.generator import (
    build_prompt,
    choice_correct,
    exact_match,
    extract_choice,
    normalise,
)
from core.ids import chunk_id, content_hash, doc_id, query_id, slug
from core.types import Chunk, Query


class TestEnvironmentPathOverrides:
    """The two variables that let one config run on Windows, Colab and Barkla."""

    def test_datasets_dir_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAGBENCH_DATASETS_DIR", str(tmp_path / "corpus"))
        assert Config().paths.datasets_dir == tmp_path / "corpus"

    def test_artifacts_override_reroots_every_generated_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAGBENCH_ARTIFACTS_DIR", str(tmp_path / "out"))
        paths = Config().paths
        # One variable must relocate all four, not just the artifacts root.
        assert paths.corpora_dir == tmp_path / "out" / "corpora"
        assert paths.indexes_dir == tmp_path / "out" / "indexes"
        assert paths.results_dir == tmp_path / "out" / "results"
        assert paths.database_path == tmp_path / "out" / "benchmark.sqlite"

    def test_unset_variables_resolve_under_project_root(self, monkeypatch):
        monkeypatch.delenv("RAGBENCH_DATASETS_DIR", raising=False)
        monkeypatch.delenv("RAGBENCH_ARTIFACTS_DIR", raising=False)
        paths = Config().paths
        assert paths.datasets_dir == PROJECT_ROOT / "datasets"
        assert paths.database_path == PROJECT_ROOT / "artifacts" / "benchmark.sqlite"

    def test_blank_variable_is_ignored(self, monkeypatch):
        """An empty variable is a common shell accident; treat it as unset."""
        monkeypatch.setenv("RAGBENCH_DATASETS_DIR", "   ")
        assert Config().paths.datasets_dir == PROJECT_ROOT / "datasets"

    def test_absolute_configured_path_beats_the_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAGBENCH_ARTIFACTS_DIR", str(tmp_path / "out"))
        config = Config()
        config.paths.corpora = str(tmp_path / "explicit")
        assert config.paths.corpora_dir == tmp_path / "explicit"


class TestConfig:
    def test_defaults_are_sane(self):
        config = Config()
        assert config.retrieval.top_k == 10
        assert config.retrieval.rrf_k == 60
        assert config.embedding.normalize is True
        assert config.generation.enabled is False

    def test_paths_resolve_relative_to_project_root(self):
        config = Config()
        assert config.paths.database_path.is_absolute()
        assert config.paths.corpora_dir.is_absolute()

    def test_absolute_paths_preserved(self, tmp_path):
        config = Config()
        config.paths.corpora = str(tmp_path / "c")
        assert config.paths.corpora_dir == tmp_path / "c"

    def test_yaml_override(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(
            "retrieval:\n  top_k: 25\n  rrf_k: 10\n"
            "embedding:\n  model_name: some/model\n"
            "datasets:\n  sciq:\n    max_queries: 7\n"
            "    chunk:\n      strategy: fixed\n      size_tokens: 128\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.retrieval.top_k == 25
        assert config.retrieval.rrf_k == 10
        assert config.embedding.model_name == "some/model"
        assert config.dataset("sciq").max_queries == 7
        assert config.dataset("sciq").chunk.size_tokens == 128
        # untouched values keep their defaults
        assert config.retrieval.bm25_k1 == 1.5

    def test_unknown_key_rejected(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("retrieval:\n  nonsense: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="unknown configuration key"):
            load_config(path)

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("does/not/exist.yaml")

    def test_shipped_default_config_loads(self):
        config = load_config("config/default.yaml")
        assert config.embedding.model_name.startswith("sentence-transformers/")
        assert config.dataset("cuad").split == "all"
        assert config.dataset("pubmedqa").split == "labeled"

    def test_dataset_chunk_override_wins(self):
        config = Config()
        default = ChunkConfig(strategy="passage")
        assert config.chunk_config_for("sciq", default) is default

    def test_fingerprints_differ_by_strategy(self):
        assert ChunkConfig(strategy="fixed", size_tokens=256, overlap_tokens=64).fingerprint() \
            != ChunkConfig(strategy="passage", max_tokens=512).fingerprint()


class TestIds:
    def test_doc_id_shape(self):
        assert doc_id("cuad", "all", "Some Contract").startswith("cuad/all/")

    def test_slug_is_filesystem_safe(self):
        assert "/" not in slug("a/b\\c:d*e")
        assert " " not in slug("hello world")

    def test_long_values_get_hash_suffix(self):
        a = slug("x" * 100 + "A")
        b = slug("x" * 100 + "B")
        assert a != b        # no collision despite the shared prefix
        assert "~" in a

    def test_chunk_ids_are_ordered_and_padded(self):
        assert chunk_id("d", 7).endswith("#c00007")

    def test_stable_across_calls(self):
        assert query_id("sciq", "test", "1") == query_id("sciq", "test", "1")
        assert content_hash("abc") == content_hash("abc")
        assert content_hash("abc") != content_hash("abd")

    def test_empty_input_does_not_produce_empty_slug(self):
        assert slug("") == "x"


class TestGenerationHelpers:
    def _query(self, **metadata):
        return Query("q", "ds", "s", "What is the answer?", metadata=metadata)

    def test_prompt_includes_context_and_question(self):
        chunks = [Chunk("c1", "d", "ds", "s", "relevant context here", 0, 0, 20)]
        prompt = build_prompt(self._query(), chunks)
        assert "relevant context here" in prompt
        assert "What is the answer?" in prompt

    def test_prompt_limits_context_chunks(self):
        chunks = [
            Chunk(f"c{i}", "d", "ds", "s", f"chunk number {i}", i, 0, 5) for i in range(10)
        ]
        prompt = build_prompt(self._query(), chunks, max_chunks=2)
        assert "chunk number 0" in prompt
        assert "chunk number 5" not in prompt

    def test_prompt_renders_options(self):
        query = self._query(options={"A": "first", "B": "second"})
        prompt = build_prompt(query, [])
        assert "A. first" in prompt and "B. second" in prompt

    def test_prompt_handles_no_context(self):
        assert "(no context retrieved)" in build_prompt(self._query(), [])

    @pytest.mark.parametrize("response,expected", [
        ("B", "B"), ("B.", "B"), ("(C)", "C"), ("The answer is D", "D"),
        ("  a) because", "A"), ("nothing relevant", None),
    ])
    def test_extract_choice(self, response, expected):
        options = {k: f"option {k}" for k in "ABCD"}
        assert extract_choice(response, options) == expected

    def test_extract_choice_by_option_text(self):
        options = {"A": "aspirin", "B": "paracetamol"}
        assert extract_choice("I would give paracetamol", options) == "B"

    def test_choice_correct(self):
        options = {"A": "x", "B": "y"}
        assert choice_correct("B", options, "B") == 1.0
        assert choice_correct("A", options, "B") == 0.0

    def test_choice_correct_null_without_options(self):
        assert choice_correct("anything", {}, None) is None

    def test_unparseable_answer_scores_zero_not_null(self):
        # The model answered; it just answered unusably. That is incorrect,
        # not unmeasurable.
        assert choice_correct("hmm", {"A": "x", "B": "y"}, "A") == 0.0

    def test_exact_match(self):
        assert exact_match("Yes", "yes") == 1.0
        assert exact_match("no", "yes") == 0.0

    def test_exact_match_null_without_reference(self):
        assert exact_match("anything", None) is None
        assert exact_match("anything", "") is None

    def test_exact_match_requires_whole_words_not_substrings(self):
        # Regression: raw substring containment finds PubMedQA's "no" inside
        # "does *no*t", scoring an explicit refusal as a correct answer.
        assert exact_match("The context does not provide an answer.", "no") == 0.0
        assert exact_match("There is no information provided.", "yes") == 0.0
        # A genuine decision still matches.
        assert exact_match("No, the evidence does not support it.", "no") == 1.0
        assert exact_match("Maybe; the evidence is mixed.", "maybe") == 1.0

    def test_exact_match_allows_reference_phrase_inside_a_longer_answer(self):
        assert exact_match("A. Spironolactone is the cause.", "Spironolactone") == 1.0

    def test_normalise_strips_punctuation(self):
        assert normalise("Yes, indeed!") == "yes  indeed"


class TestDeclaredDependencies:
    """Every third-party import must be installable from requirements.txt.

    scipy was imported by ``evaluation.significance`` for months without being
    declared. It happened to be present as a transitive dependency of
    sentence-transformers, so the suite passed and a fresh
    ``pip install -r requirements.txt`` followed by ``cli significance`` did
    not. A benchmark that claims reproducibility has to be installable from its
    own manifest.
    """

    # Import name -> distribution name, where they differ by more than the
    # hyphen/underscore equivalence that _normalise already handles.
    ALIASES = {
        "yaml": "pyyaml",
        "faiss": "faiss-cpu",
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "dateutil": "python-dateutil",
    }

    @staticmethod
    def _normalise(name):
        """PEP 503: rank_bm25 and rank-bm25 are the same distribution."""
        return re.sub(r"[-_.]+", "-", name).lower()

    @classmethod
    def _declared(cls):
        text = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        names = set()
        for line in text.splitlines():
            line = line.split("#")[0].strip()
            if line:
                names.add(cls._normalise(re.split(r"[=<>~\[!]", line)[0].strip()))
        return names

    @staticmethod
    def _third_party_imports():
        import ast
        import sys

        stdlib = set(sys.stdlib_module_names)
        packages = {p.name for p in PROJECT_ROOT.iterdir() if p.is_dir()} | {"cli"}
        found = {}
        for path in PROJECT_ROOT.rglob("*.py"):
            parts = set(path.parts)
            if parts & {"venv", ".venv", "__pycache__", "build", "dist"}:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split(".")[0]]
                else:
                    continue
                for module in modules:
                    if module not in stdlib and module not in packages:
                        found.setdefault(module, path.relative_to(PROJECT_ROOT))
        return found

    def test_every_import_is_declared(self):
        declared = self._declared()
        undeclared = {
            module: where
            for module, where in self._third_party_imports().items()
            if self._normalise(module) not in declared
            and self._normalise(self.ALIASES.get(module, "")) not in declared
        }
        assert not undeclared, (
            "imported but missing from requirements.txt: "
            + ", ".join(f"{m} (in {p})" for m, p in sorted(undeclared.items()))
        )

    def test_requirements_and_pyproject_agree(self):
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
        pyproject = {
            self._normalise(re.split(r"[=<>~\[!]", item.strip().strip('",'))[0].strip())
            for item in block.splitlines() if item.strip().startswith('"')
        }
        missing = pyproject - self._declared()
        assert not missing, f"in pyproject.toml but not requirements.txt: {sorted(missing)}"
