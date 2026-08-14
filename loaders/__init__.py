"""Dataset adapter registry."""

from __future__ import annotations

from typing import Dict, List, Type

from core.settings import Config
from .base import DatasetAdapter, LoadedSplit
from .casehold import CaseHOLDAdapter
from .cuad import CUADAdapter
from .medqa import MedQAAdapter
from .pubmedqa import PubMedQAAdapter
from .qasper import QasperAdapter
from .sciq import SciQAdapter

ADAPTERS: Dict[str, Type[DatasetAdapter]] = {
    cls.spec.name: cls
    for cls in (
        CUADAdapter,
        CaseHOLDAdapter,
        MedQAAdapter,
        PubMedQAAdapter,
        QasperAdapter,
        SciQAdapter,
    )
}

DOMAINS: Dict[str, List[str]] = {}
for _name, _cls in ADAPTERS.items():
    DOMAINS.setdefault(_cls.spec.domain, []).append(_name)


def get_adapter(name: str, config: Config) -> DatasetAdapter:
    try:
        cls = ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown dataset {name!r}; available: {', '.join(sorted(ADAPTERS))}"
        ) from None
    return cls(config)


def all_dataset_names() -> List[str]:
    return sorted(ADAPTERS)


def retrieval_capable_datasets() -> List[str]:
    """Datasets whose ground truth supports retrieval metrics (excludes MedQA)."""
    return sorted(n for n, c in ADAPTERS.items() if c.spec.supports_retrieval_metrics)


__all__ = [
    "ADAPTERS", "DOMAINS", "DatasetAdapter", "LoadedSplit",
    "get_adapter", "all_dataset_names", "retrieval_capable_datasets",
]
