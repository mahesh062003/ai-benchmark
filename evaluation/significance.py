"""Paired significance testing between retrieval strategies.

The aggregate table reports point estimates: hybrid scored 0.6158 on CUAD,
dense 0.6094. On its own that ordering is not a result, because it carries no
statement about whether the gap would survive a different sample of queries.
Some gaps in this benchmark are 0.0007 wide. This module decides which
differences are real.

Design decisions, each of which is defensible in a viva:

Pairing on query id, not comparing means
    Both strategies answer the *same* queries, so the comparison is paired.
    A paired test removes between-query variance -- the fact that some queries
    are simply harder than others -- and is far more powerful than treating the
    two score sets as independent samples.

Wilcoxon signed-rank rather than a paired t-test
    Retrieval metrics are bounded in [0, 1], heavily skewed, and zero-inflated:
    a large share of queries score exactly 0 or exactly 1. That violates the
    normality assumption behind a t-test. Wilcoxon assumes only that the
    differences are symmetrically distributed about the median.

Bootstrap confidence interval alongside the p-value
    A p-value says whether a difference exists; it says nothing about size.
    The bootstrap interval gives the effect size in the metric's own units,
    which is what a reader actually needs. It is computed by resampling
    *query pairs*, preserving the pairing.

NULL is excluded, never imputed
    A query that is unscoreable under either strategy is dropped from that
    comparison rather than entered as 0.0. Imputing zero would manufacture
    agreement between strategies on queries that were never measured, and
    would silently inflate the sample size.

Holm-Bonferroni correction
    Three strategies give three pairwise comparisons per dataset and metric.
    Testing all of them at p < 0.05 inflates the family-wise error rate. Holm
    controls it while being uniformly more powerful than plain Bonferroni.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260814
DEFAULT_ALPHA = 0.05


@dataclass
class Comparison:
    """One paired test between two strategies on one dataset and metric."""

    dataset: str
    metric: str
    method_a: str
    method_b: str
    n_pairs: int
    n_differing: int
    mean_a: float
    mean_b: float
    mean_diff: float          # method_a minus method_b
    ci_low: float
    ci_high: float
    p_value: float
    p_corrected: float = float("nan")
    alpha: float = DEFAULT_ALPHA

    @property
    def significant(self) -> bool:
        return self.p_corrected == self.p_corrected and self.p_corrected < self.alpha

    @property
    def winner(self) -> Optional[str]:
        """The better strategy, or None when the difference is not significant."""
        if not self.significant:
            return None
        return self.method_a if self.mean_diff > 0 else self.method_b


def load_per_query(
    connection: sqlite3.Connection,
    run_id: str,
    dataset: str,
    method: str,
    metric: str,
    k: int = 10,
) -> Dict[str, float]:
    """Per-query scores for one strategy, keyed by query id.

    Only rows the benchmark marked scoreable are returned, so a dataset with
    no retrieval ground truth yields an empty mapping and is skipped upstream
    rather than being compared against zeros.
    """
    rows = connection.execute(
        "SELECT query_id, mrr, metrics_json FROM query_metrics"
        " WHERE run_id = ? AND dataset = ? AND method = ? AND scoreable = 1",
        (run_id, dataset, method),
    )
    scores: Dict[str, float] = {}
    for query_id, mrr, payload in rows:
        if metric == "mrr":
            value = mrr
        else:
            parsed = json.loads(payload) if payload else {}
            value = (parsed.get(metric) or {}).get(str(k))
        if value is not None:
            scores[query_id] = float(value)
    return scores


def align(a: Dict[str, float], b: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return score arrays over the queries both strategies scored, in a stable order."""
    shared = sorted(set(a) & set(b))
    return (
        np.array([a[q] for q in shared], dtype=float),
        np.array([b[q] for q in shared], dtype=float),
    )


def bootstrap_ci(
    differences: np.ndarray,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Tuple[float, float]:
    """Percentile bootstrap interval for the mean paired difference.

    Query *pairs* are resampled, not the two score sets independently, so the
    pairing that makes the test powerful is preserved in every resample.
    """
    if differences.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value for paired scores.

    Where every query scores identically under both strategies there is no
    evidence of a difference, and scipy raises rather than returning 1.0; that
    case is reported as p = 1.0, which is what it means.
    """
    from scipy.stats import wilcoxon

    differences = a - b
    if differences.size == 0 or np.all(differences == 0):
        return 1.0
    try:
        return float(wilcoxon(a, b, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values, preserving input order.

    Adjusted values are made monotone non-decreasing in rank order so that a
    later comparison can never appear more significant than a stronger one.
    """
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, index in enumerate(order):
        scaled = (n - rank) * p_values[index]
        running = max(running, scaled)
        adjusted[index] = min(1.0, running)
    return adjusted


def compare_pair(
    connection: sqlite3.Connection,
    run_id: str,
    dataset: str,
    metric: str,
    method_a: str,
    method_b: str,
    k: int = 10,
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> Optional[Comparison]:
    """Test one strategy pair, or None when they share no scoreable queries."""
    a_scores = load_per_query(connection, run_id, dataset, method_a, metric, k)
    b_scores = load_per_query(connection, run_id, dataset, method_b, metric, k)
    a, b = align(a_scores, b_scores)
    if a.size == 0:
        return None

    differences = a - b
    lo, hi = bootstrap_ci(differences, resamples, alpha, seed)
    return Comparison(
        dataset=dataset,
        metric=metric if metric == "mrr" else f"{metric}@{k}",
        method_a=method_a,
        method_b=method_b,
        n_pairs=int(a.size),
        n_differing=int(np.count_nonzero(differences)),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_diff=float(differences.mean()),
        ci_low=lo,
        ci_high=hi,
        p_value=wilcoxon_p(a, b),
        alpha=alpha,
    )


def run_comparisons(
    connection: sqlite3.Connection,
    run_id: str,
    datasets: Sequence[str],
    methods: Sequence[str],
    metrics: Sequence[Tuple[str, int]],
    resamples: int = DEFAULT_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = DEFAULT_SEED,
) -> List[Comparison]:
    """All pairwise tests, Holm-corrected within each dataset and metric.

    The correction family is one dataset and one metric -- the three strategy
    pairs a reader compares when reading a single row of the results table.
    Correcting across datasets as well would be defensible but far more
    conservative, and would answer a question nobody asks.
    """
    results: List[Comparison] = []
    for dataset in datasets:
        for metric, k in metrics:
            family: List[Comparison] = []
            for method_a, method_b in combinations(methods, 2):
                comparison = compare_pair(
                    connection, run_id, dataset, metric, method_a, method_b,
                    k=k, resamples=resamples, alpha=alpha, seed=seed,
                )
                if comparison is not None:
                    family.append(comparison)
            for comparison, adjusted in zip(family, holm_bonferroni([c.p_value for c in family])):
                comparison.p_corrected = adjusted
            results.extend(family)
    return results
