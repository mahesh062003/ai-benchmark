"""The adapter contract shared by all six datasets."""

from __future__ import annotations

import abc
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from core.settings import ChunkConfig, Config
from core.logging_setup import SkipLog
from core.types import DatasetSpec, Query, SourceDocument

OPTION_KEYS = "ABCDEFGH"


def build_options(
    correct: str, distractors: Sequence[str], seed: str
) -> Tuple[Dict[str, str], Optional[str]]:
    """Label a correct answer and its distractors as a multiple-choice set.

    The generation stage renders ``options`` into the prompt and scores the
    model's chosen letter against ``answer_idx``. Both live in query metadata,
    so a dataset that supplies them gets choice accuracy and one that does not
    leaves it NULL.

    Candidates are shuffled rather than left in source order, because source
    order always puts the correct answer first and a model that always replied
    "A" would score 100%. The shuffle is seeded per query so the order is
    identical on every run: the frozen task sets that let several models answer
    byte-identical prompts depend on it.

    Returns ``({}, None)`` when there is no usable answer, which keeps choice
    accuracy NULL rather than scoring an unanswerable question as wrong.
    """
    correct = (correct or "").strip()
    candidates = [c for c in (correct, *distractors) if (c or "").strip()]
    if not correct or len(candidates) < 2:
        return {}, None
    candidates = candidates[: len(OPTION_KEYS)]

    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    options = {OPTION_KEYS[i]: text for i, text in enumerate(shuffled)}
    gold_key = next(key for key, text in options.items() if text == correct)
    return options, gold_key


@dataclass
class LoadedSplit:
    """What an adapter returns for one split.

    ``documents`` are already de-duplicated; ``queries`` reference them by id.
    """

    documents: List[SourceDocument]
    queries: List[Query]
    skips: SkipLog
    stats: Dict[str, int] = field(default_factory=dict)


class DatasetAdapter(abc.ABC):
    """Normalises one dataset onto the shared data model.

    An adapter is responsible for exactly three things:
      1. reading the raw files as they actually ship,
      2. emitting de-duplicated SourceDocuments and Queries,
      3. expressing the dataset's ground truth as EvidenceSpans -- or declaring
         that none exists.

    An adapter must never invent evidence. Where a record has no usable
    evidence it emits the query with an empty evidence list, which excludes it
    from retrieval metrics.
    """

    spec: DatasetSpec
    default_chunk: ChunkConfig = ChunkConfig()

    def __init__(self, config: Config) -> None:
        self.config = config
        self.root = config.paths.datasets_dir

    @property
    def name(self) -> str:
        return self.spec.name

    def chunk_config(self) -> ChunkConfig:
        return self.config.chunk_config_for(self.name, self.default_chunk)

    def default_split(self) -> str:
        configured = self.config.dataset(self.name).split
        return configured or self.spec.splits[-1]

    @abc.abstractmethod
    def load(self, split: str) -> LoadedSplit:
        """Load one split. Raise FileNotFoundError if the data is absent."""

    def _require(self, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: expected dataset file missing: {path}\n"
                f"Check paths.datasets in your config (currently {self.root})."
            )
        return path
