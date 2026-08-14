"""Configuration.

All tunable values live here or in a YAML file that overrides these defaults.
Paths are resolved relative to the project root, so nothing is machine-specific
and commands work from the repo root.

Two environment variables relocate the two directories that genuinely differ
between machines, so the same YAML runs unchanged on Windows, Google Colab and
the Barkla HPC cluster:

``RAGBENCH_DATASETS_DIR``
    Where the raw datasets live. Set this when the corpus is mounted outside
    the repository, for example on Google Drive or scratch storage.

``RAGBENCH_ARTIFACTS_DIR``
    Where generated corpora, indexes, results and the database are written.
    Point this at persistent storage so a run survives a Colab disconnect.

Both take precedence over the YAML file, which in turn overrides the defaults
below. Neither is required: with both unset, everything resolves under the
repository root exactly as before.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ENV_DATASETS_DIR = "RAGBENCH_DATASETS_DIR"
ENV_ARTIFACTS_DIR = "RAGBENCH_ARTIFACTS_DIR"


def _env_dir(name: str) -> Optional[Path]:
    """Return an environment path override, or None when unset or blank."""
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass
class PathConfig:
    datasets: str = "datasets"
    artifacts: str = "artifacts"
    corpora: str = "artifacts/corpora"
    indexes: str = "artifacts/indexes"
    results: str = "artifacts/results"
    database: str = "artifacts/benchmark.sqlite"

    def resolve(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p

    def _under_artifacts(self, configured: str) -> Path:
        """Resolve a path that normally sits inside the artifacts directory.

        When ``RAGBENCH_ARTIFACTS_DIR`` is set, the configured value is
        re-rooted onto it, so one variable relocates every generated file
        rather than four. An absolute configured path always wins, on the
        assumption that it was set deliberately.
        """
        override = _env_dir(ENV_ARTIFACTS_DIR)
        configured_path = Path(configured).expanduser()
        if override is None or configured_path.is_absolute():
            return self.resolve(configured)
        try:
            tail = configured_path.relative_to(Path(self.artifacts))
        except ValueError:
            # Configured outside the artifacts tree; keep only its name.
            tail = Path(configured_path.name)
        return override / tail

    @property
    def datasets_dir(self) -> Path:
        override = _env_dir(ENV_DATASETS_DIR)
        return override if override is not None else self.resolve(self.datasets)

    @property
    def artifacts_dir(self) -> Path:
        override = _env_dir(ENV_ARTIFACTS_DIR)
        return override if override is not None else self.resolve(self.artifacts)

    @property
    def corpora_dir(self) -> Path:
        return self._under_artifacts(self.corpora)

    @property
    def indexes_dir(self) -> Path:
        return self._under_artifacts(self.indexes)

    @property
    def results_dir(self) -> Path:
        return self._under_artifacts(self.results)

    @property
    def database_path(self) -> Path:
        return self._under_artifacts(self.database)


@dataclass
class ChunkConfig:
    """Chunking parameters.

    ``strategy``:
      ``passage`` -- one chunk per source document, split only if it exceeds
                     ``max_tokens``. Correct where the dataset's own unit is
                     already a passage (QASPER paragraphs, PubMedQA abstract
                     sections, SciQ supports, CaseHOLD holdings).
      ``fixed``   -- sliding window of ``size_tokens`` whitespace tokens with
                     ``overlap_tokens`` of overlap. Used for long free text
                     (CUAD contracts, MedQA textbooks).

    Defaults are justified in docs/METHODOLOGY.md rather than copied from
    generic RAG tutorials: 256 tokens is close to the median CUAD clause span
    plus surrounding context, and sits inside the 256-token input limit of the
    default MiniLM embedding model so no chunk is silently truncated.
    """

    strategy: str = "fixed"
    size_tokens: int = 256
    overlap_tokens: int = 64
    max_tokens: int = 512
    min_chunk_chars: int = 20

    def fingerprint(self) -> str:
        if self.strategy == "passage":
            return f"passage-max{self.max_tokens}"
        return f"fixed-{self.size_tokens}-{self.overlap_tokens}"


@dataclass
class RetrievalConfig:
    top_k: int = 10
    metric_ks: List[int] = field(default_factory=lambda: [1, 5, 10])
    methods: List[str] = field(default_factory=lambda: ["bm25", "dense", "hybrid"])
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    rrf_k: int = 60
    rrf_candidate_depth: int = 100
    """How deep each base ranker is consulted before fusion. Larger than top_k
    so that RRF can promote items outside either list's head."""


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 64
    device: Optional[str] = None  # None -> auto (cuda if available)
    normalize: bool = True
    """Embeddings are L2-normalised so that FAISS inner product == cosine."""


@dataclass
class GenerationConfig:
    """The optional generation stage.

    ``enabled`` defaults to False and nothing in this section is touched by the
    retrieval pipeline, so the benchmark runs unchanged on a machine with no
    Ollama installed.
    """

    enabled: bool = False
    model: str = "llama3.1"
    """Model used by the single-model ``generate`` command."""

    models: List[str] = field(
        default_factory=lambda: ["llama3.1", "gemma2", "mistral", "qwen2.5"]
    )
    """Models compared by ``generate-all``. Each answers identical inputs."""

    host: str = "http://localhost:11434"
    temperature: float = 0.0
    max_context_chunks: int = 5
    timeout_seconds: int = 120
    num_predict: int = 512
    queries_per_dataset: int = 50
    """Seeded sample size per dataset, per method, per model."""


@dataclass
class ScoringConfig:
    """Answer-grounding measures applied after generation."""

    judge_model: str = "llama3.1"
    """One fixed LLM judge rates every model's answers. Holding the judge
    constant is what makes faithfulness comparable across models; a model
    judging its own output would not be on the same footing as its rivals."""

    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    entail_threshold: float = 0.5
    """Entailment probability at or above which a sentence counts as grounded."""

    contradict_threshold: float = 0.5
    """Contradiction probability required before a sentence is called a
    contradiction rather than merely unsupported."""

    nli_batch_size: int = 32
    nli_max_length: int = 512

    faithfulness_sample: Optional[int] = None
    """Cap on how many answers are judged for faithfulness, drawn as a
    stratified sample across every (dataset, method, model) cell. Two judge
    calls per answer make full coverage the most expensive measure here, so a
    subsample keeps every comparison cell represented at a fraction of the
    cost. ``None`` scores every answer. Unsampled rows keep NULL, which already
    means 'not measured' everywhere else in the schema."""


@dataclass
class DatasetConfig:
    """Per-dataset overrides. ``max_queries`` supports fast smoke runs."""

    enabled: bool = True
    split: Optional[str] = None
    max_queries: Optional[int] = None
    max_documents: Optional[int] = None
    chunk: Optional[ChunkConfig] = None


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)
    seed: int = 42
    log_level: str = "INFO"

    def dataset(self, name: str) -> DatasetConfig:
        return self.datasets.get(name, DatasetConfig())

    def chunk_config_for(self, name: str, default: ChunkConfig) -> ChunkConfig:
        """Resolve chunking for a dataset: explicit override > adapter default."""
        override = self.dataset(name).chunk
        return override if override is not None else default

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _merge(base: Any, override: Dict[str, Any]) -> Any:
    """Recursively apply a plain dict onto a dataclass instance."""
    for key, value in override.items():
        if not hasattr(base, key):
            raise ValueError(f"unknown configuration key: {key}")
        current = getattr(base, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
        else:
            setattr(base, key, value)
    return base


def load_config(path: Optional[Path | str] = None) -> Config:
    """Load configuration, optionally overlaying a YAML file."""
    cfg = Config()
    if path is None:
        return cfg
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    dataset_raw = raw.pop("datasets", {}) or {}
    _merge(cfg, raw)
    for name, values in dataset_raw.items():
        dc = DatasetConfig()
        chunk_raw = (values or {}).pop("chunk", None)
        _merge(dc, values or {})
        if chunk_raw:
            dc.chunk = _merge(ChunkConfig(), chunk_raw)
        cfg.datasets[name] = dc
    return cfg
