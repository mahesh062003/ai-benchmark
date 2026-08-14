"""The adapter contract shared by all six datasets."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from core.settings import ChunkConfig, Config
from core.logging_setup import SkipLog
from core.types import DatasetSpec, Query, SourceDocument


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
