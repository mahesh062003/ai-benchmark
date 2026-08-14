"""Logging configuration and skip accounting.

Silently dropping records is the easiest way to produce a wrong benchmark, so
every adapter reports what it discarded and why through ``SkipLog``.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict

from rich.logging import RichHandler

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    root = logging.getLogger("ragbench")
    root.setLevel(level.upper())
    if not _CONFIGURED:
        handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)
        _CONFIGURED = True
    else:
        for h in root.handlers:
            h.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"ragbench.{name}")


@dataclass
class SkipLog:
    """Counts records skipped during loading, grouped by reason."""

    dataset: str
    split: str
    counts: Counter = field(default_factory=Counter)

    def skip(self, reason: str, n: int = 1) -> None:
        self.counts[reason] += n

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def as_dict(self) -> Dict[str, int]:
        return dict(self.counts)

    def report(self, logger: logging.Logger, kept: int) -> None:
        if not self.counts:
            logger.info("%s/%s: kept %d records, skipped none", self.dataset, self.split, kept)
            return
        detail = ", ".join(f"{reason}={n}" for reason, n in sorted(self.counts.items()))
        logger.warning(
            "%s/%s: kept %d records, skipped %d (%s)",
            self.dataset, self.split, kept, self.total, detail,
        )
