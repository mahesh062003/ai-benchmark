"""Streamlit dashboard for the RAG benchmark.

Reads the SQLite results store and presents retrieval and generation side by
side. Two properties are load-bearing and survive any restyling:

Generation is optional
    The retrieval half renders from the database alone and never imports the
    generation stack or contacts Ollama. When no answers have been generated,
    the generation half degrades to an explanatory panel instead of an error,
    so a machine with no Ollama still gets the full retrieval view.

NULL is not zero
    A metric that could not legitimately be computed is missing from the chart
    and shown as an em dash in the table, never plotted as 0.0. MedQA has no
    retrieval ground truth, so it contributes no retrieval bar at all;
    conflating that with a score of zero would invent a result.

Run with:  ragbench dashboard      (or: streamlit run dashboard/app.py)
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.settings import load_config  # noqa: E402
from loaders import ADAPTERS  # noqa: E402

# Domain comes from each adapter's declared spec, not from whichever retrieval
# rows survived the filter, so generation still groups by domain when a dataset
# contributes answers but no scoreable retrieval metrics.
DATASET_DOMAIN = {name: adapter.spec.domain for name, adapter in ADAPTERS.items()}

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------

BG = "#0F1117"
SURFACE = "#161A23"
SURFACE_RAISED = "#1C2130"
BORDER = "#282E3C"
TEXT = "#EDF0F6"
MUTED = "#868FA5"
GOLD = "#F0A500"
GOLD_SOFT = "#FFC24B"
BLUE = "#5B8DEF"
BLUE_DEEP = "#3B63C4"
GRID = "#222736"

# Retrieval reads blue, generation reads gold, so the two halves of the page
# stay distinguishable at a glance even when charts are viewed side by side.
RETRIEVAL_COLOURS = [BLUE, "#4A79DC", BLUE_DEEP]
GENERATION_COLOURS = [GOLD, "#C98600"]

DOMAIN_COLOURS = {
    "legal": "#3B82F6",
    "medical": "#10B981",
    "scientific": "#A855F7",
}
DOMAIN_ORDER = ["legal", "medical", "scientific"]

# The database stores the retriever's internal name; the UI shows the name a
# reader recognises from the literature.
METHOD_LABEL = {"bm25": "BM25", "dense": "FAISS", "hybrid": "HYBRID"}
METHOD_ORDER = ["bm25", "dense", "hybrid"]

CSS = f"""
<style>
  .stApp {{ background: {BG}; }}
  header[data-testid="stHeader"] {{ background: transparent; }}
  .block-container {{ padding: 0 2.2rem 3rem; max-width: 1500px; }}

  html, body, [class*="css"] {{
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
    color: {TEXT};
  }}

  /* ---------- header band ---------- */
  .rb-header {{
    background: linear-gradient(105deg, #10131C 0%, #161B29 55%, #1A2033 100%);
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1.5rem 1.9rem;
    margin: 1.1rem 0 1.4rem;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 1.2rem;
  }}
  .rb-eyebrow {{
    color: {GOLD}; font-size: .68rem; letter-spacing: .22em;
    text-transform: uppercase; font-weight: 600; margin-bottom: .5rem;
  }}
  .rb-eyebrow::before {{ content: "— "; opacity: .7; }}
  .rb-title {{ font-size: 1.95rem; font-weight: 700; line-height: 1.15; }}
  .rb-title em {{ color: {GOLD}; font-style: italic; font-weight: 600; }}
  .rb-sub {{ color: {MUTED}; font-size: .84rem; margin-top: .45rem; letter-spacing: .02em; }}

  .rb-stats {{ display: flex; gap: 2.6rem; }}
  .rb-stat {{ text-align: right; }}
  .rb-stat-value {{ font-size: 1.55rem; font-weight: 700; line-height: 1.1; }}
  .rb-stat-label {{
    color: {MUTED}; font-size: .61rem; letter-spacing: .17em;
    text-transform: uppercase; margin-top: .35rem;
  }}

  /* ---------- cards ---------- */
  .rb-card {{
    background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 13px; padding: 1.15rem 1.3rem 0.6rem;
  }}
  .rb-card-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: .2rem;
  }}
  .rb-card-title {{ font-size: 1.02rem; font-weight: 650; }}
  .rb-card-note {{ color: {MUTED}; font-size: .69rem; letter-spacing: .05em; }}
  .rb-legend {{
    color: {MUTED}; font-size: .72rem; padding: .1rem 0 .8rem;
    display: flex; align-items: center; gap: .45rem;
  }}
  .rb-dot {{ width: .55rem; height: .55rem; border-radius: 50%; display: inline-block; }}

  /* ---------- insight ---------- */
  .rb-insight {{
    background: {SURFACE_RAISED}; border-left: 3px solid {GOLD};
    border-radius: 7px; padding: .85rem .95rem; margin-top: .9rem;
  }}
  .rb-insight-label {{
    color: {GOLD}; font-size: .6rem; letter-spacing: .18em;
    text-transform: uppercase; font-weight: 600; margin-bottom: .4rem;
  }}
  .rb-insight-body {{ font-size: .8rem; line-height: 1.5; color: #C9D1E2; }}
  .rb-insight-body b {{ color: {TEXT}; }}

  /* ---------- badges ---------- */
  .rb-badge {{
    display: inline-flex; align-items: center; gap: .38rem;
    padding: .16rem .55rem .16rem .45rem; border-radius: 5px;
    font-size: .72rem; font-weight: 550; text-transform: capitalize;
  }}
  .rb-tag {{
    display: inline-block; padding: .16rem .55rem; border-radius: 5px;
    font-size: .68rem; font-weight: 600; letter-spacing: .06em;
    background: rgba(91,141,239,.13); color: #9DBBFF;
    border: 1px solid rgba(91,141,239,.28);
  }}

  /* ---------- results table ---------- */
  .rb-table {{ width: 100%; border-collapse: collapse; margin-top: .35rem; }}
  .rb-table th {{
    text-align: left; color: {MUTED}; font-size: .62rem; font-weight: 600;
    letter-spacing: .15em; text-transform: uppercase;
    padding: .55rem .7rem; border-bottom: 1px solid {BORDER};
  }}
  .rb-table td {{
    padding: .55rem .7rem; font-size: .82rem;
    border-bottom: 1px solid rgba(40,46,60,.55); color: #D6DCEA;
  }}
  .rb-table tr:hover td {{ background: rgba(91,141,239,.05); }}
  .rb-num {{ font-variant-numeric: tabular-nums; }}
  .rb-null-cell {{ color: #5A6275; }}
  .rb-good {{ color: #34D399; font-weight: 600; }}
  .rb-bad {{ color: #F87171; font-weight: 600; }}

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {{
    background: {SURFACE}; border-right: 1px solid {BORDER};
  }}
  section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}
  .rb-filter-label {{
    color: {GOLD}; font-size: .62rem; letter-spacing: .16em;
    text-transform: uppercase; font-weight: 600; margin: .9rem 0 .1rem;
  }}
  .rb-filter-label::before {{ content: "▾ "; opacity: .75; }}

  div[data-baseweb="select"] > div {{
    background: {SURFACE_RAISED} !important;
    border-color: {BORDER} !important; border-radius: 7px !important;
    font-size: .82rem;
  }}
  .stButton > button {{
    width: 100%; background: {SURFACE_RAISED}; color: {TEXT};
    border: 1px solid {BORDER}; border-radius: 7px;
    font-size: .78rem; padding: .42rem 0;
  }}
  .stButton > button:hover {{ border-color: {GOLD}; color: {GOLD}; }}

  .rb-empty {{
    background: {SURFACE}; border: 1px dashed {BORDER}; border-radius: 11px;
    padding: 2rem 1.4rem; text-align: center; color: {MUTED}; font-size: .82rem;
    line-height: 1.6;
  }}
  .rb-empty b {{ color: {TEXT}; }}
  #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def read_tables(db_path: str, mtime: float):
    """Load the two result tables. ``mtime`` busts the cache when the run changes."""
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        aggregates = pd.read_sql_query(
            "SELECT run_id, dataset, domain, method, relevance_class, n_queries,"
            "       n_scoreable, coverage, mrr, metrics_json FROM aggregate_metrics",
            connection,
        )
        try:
            generations = pd.read_sql_query(
                "SELECT run_id, dataset, method, model, faithfulness, hallucination"
                "  FROM generations",
                connection,
            )
        except pd.errors.DatabaseError:
            generations = pd.DataFrame()
    return aggregates, generations


def expand_retrieval(aggregates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Flatten ``metrics_json`` into recall/ndcg columns at a chosen cut-off.

    Rows whose relevance class is UNSUPPORTED carry no metrics at all. They are
    kept, with NaN scores, so the table can still report that the dataset ran
    and was deliberately not scored.
    """
    if aggregates.empty:
        return aggregates.assign(recall=[], ndcg=[])

    rows = []
    for record in aggregates.to_dict("records"):
        payload = json.loads(record["metrics_json"]) if record["metrics_json"] else {}
        recall = (payload.get("recall") or {}).get(str(k))
        ndcg = (payload.get("ndcg") or {}).get(str(k))
        record["recall"] = recall
        record["ndcg"] = ndcg
        record["domain"] = record.get("domain") or DATASET_DOMAIN.get(record["dataset"], "unknown")
        rows.append(record)
    return pd.DataFrame(rows)


def available_ks(aggregates: pd.DataFrame) -> list[int]:
    ks: set[int] = set()
    for payload in aggregates.get("metrics_json", pd.Series(dtype=str)).dropna():
        recall = (json.loads(payload) or {}).get("recall") or {}
        ks |= {int(k) for k in recall}
    return sorted(ks) or [10]


def weighted_mean(frame: pd.DataFrame, value: str, weight: str) -> Optional[float]:
    """Mean weighted by scoreable queries, ignoring rows with no score.

    Weighting matters: a dataset contributing 20,000 scoreable queries should
    not carry the same influence as one contributing 40.
    """
    usable = frame[frame[value].notna() & frame[weight].notna() & (frame[weight] > 0)]
    if usable.empty:
        return None
    total = usable[weight].sum()
    if total <= 0:
        return None
    return float((usable[value] * usable[weight]).sum() / total)


def simple_mean(series: pd.Series) -> Optional[float]:
    values = series.dropna()
    return float(values.mean()) if not values.empty else None


def stars(p: float) -> str:
    """Conventional significance markers, so a chart reads at a glance."""
    if p is None or pd.isna(p):
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


@st.cache_data(show_spinner="Running paired significance tests…")
def load_significance(db_path: str, mtime: float, resamples: int = 2000) -> pd.DataFrame:
    """Paired Wilcoxon tests between every strategy pair, Holm-corrected.

    Recomputed from per-query scores rather than read from the exported CSV, so
    the dashboard can never show significance that disagrees with the database
    it is displaying. Fewer bootstrap resamples than the CLI default keeps the
    first page load responsive; the interval is a little coarser, the p-value
    is unaffected because it comes from Wilcoxon, not the bootstrap.
    """
    import sqlite3 as _sqlite3

    from evaluation.significance import run_comparisons

    connection = _sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        run_id = connection.execute(
            "SELECT run_id FROM runs ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        if run_id is None:
            return pd.DataFrame()
        datasets = [
            r[0] for r in connection.execute(
                "SELECT DISTINCT dataset FROM query_metrics WHERE run_id = ? AND scoreable = 1"
                " ORDER BY dataset", (run_id[0],)
            )
        ]
        methods = [
            r[0] for r in connection.execute(
                "SELECT DISTINCT method FROM query_metrics WHERE run_id = ? ORDER BY method",
                (run_id[0],)
            )
        ]
        comparisons = run_comparisons(
            connection, run_id[0], datasets, methods,
            [("recall", 10), ("mrr", 10), ("ndcg", 10)], resamples=resamples,
        )
    finally:
        connection.close()

    return pd.DataFrame([
        {
            "dataset": c.dataset, "metric": c.metric,
            "method_a": c.method_a, "method_b": c.method_b,
            "n": c.n_pairs, "mean_a": c.mean_a, "mean_b": c.mean_b,
            "diff": c.mean_diff, "ci_low": c.ci_low, "ci_high": c.ci_high,
            "p": c.p_corrected, "significant": c.significant,
            "winner": c.winner or "",
        }
        for c in comparisons
    ])


METRIC_KEY = {"Recall@K": "recall@10", "MRR": "mrr", "NDCG": "ndcg@10"}


def generation_summary(generations: pd.DataFrame) -> pd.DataFrame:
    """Mean faithfulness and hallucination per dataset and method.

    Unscored answers are NaN rather than 0.0, so a sweep that generated answers
    but has not run ``score-answers`` yet reports nothing instead of a
    fabricated floor.
    """
    if generations.empty:
        return pd.DataFrame(columns=["dataset", "method", "domain", "faithfulness", "hallucination"])
    grouped = (
        generations.groupby(["dataset", "method"], as_index=False)
        .agg(faithfulness=("faithfulness", "mean"), hallucination=("hallucination", "mean"))
    )
    grouped["domain"] = grouped["dataset"].map(DATASET_DOMAIN).fillna("unknown")
    return grouped


# ---------------------------------------------------------------------------
# insight
# ---------------------------------------------------------------------------


def leading_strategy(frame: pd.DataFrame, column: str, weight: str, lower_is_better=False):
    """Return (method, score) for the best strategy on one column, or None."""
    if frame.empty or column not in frame:
        return None
    scores = {}
    for method, block in frame.groupby("method"):
        value = weighted_mean(block, column, weight) if weight in block else simple_mean(block[column])
        if value is not None:
            scores[method] = value
    if not scores:
        return None
    best = min(scores, key=scores.get) if lower_is_better else max(scores, key=scores.get)
    return best, scores[best]


def build_insight(retrieval: pd.DataFrame, generation: pd.DataFrame, focus: str) -> str:
    """One sentence naming the strategy that leads, and how broadly it leads.

    The claim is deliberately scoped to the domains actually present after
    filtering: "every domain tested" means the ones on screen, not a claim
    about domains this run never touched.
    """
    faith = generation[generation["faithfulness"].notna()] if not generation.empty else pd.DataFrame()

    # Prefer a generation-based finding when scored answers exist, since
    # faithfulness is the more interesting claim; fall back to retrieval.
    if not faith.empty:
        overall = leading_strategy(faith, "faithfulness", weight="__none__")
        if overall:
            method, score = overall
            domains = faith["domain"].nunique()
            wins = sum(
                (leading_strategy(block, "faithfulness", "__none__") or (None,))[0] == method
                for _, block in faith.groupby("domain")
            )
            scope = (
                "across every domain tested"
                if wins == domains and domains > 1
                else f"in {wins} of {domains} domains"
            )
            return (
                f"<b>{METHOD_LABEL.get(method, method)}</b> leads on faithfulness "
                f"{scope}, averaging <b>{score:.2f}</b>."
            )

    column = {"Recall@K": "recall", "MRR": "mrr", "NDCG": "ndcg"}.get(focus, "recall")
    scoreable = retrieval[retrieval[column].notna()] if column in retrieval else pd.DataFrame()
    if scoreable.empty:
        return "No scoreable retrieval metrics under the current filters."

    overall = leading_strategy(scoreable, column, "n_scoreable")
    if not overall:
        return "No scoreable retrieval metrics under the current filters."
    method, score = overall
    domains = scoreable["domain"].nunique()
    wins = sum(
        (leading_strategy(block, column, "n_scoreable") or (None,))[0] == method
        for _, block in scoreable.groupby("domain")
    )
    scope = (
        "across every domain tested"
        if wins == domains and domains > 1
        else f"in {wins} of {domains} domains"
    )
    label = {"recall": "Recall", "mrr": "MRR", "ndcg": "NDCG"}[column]
    return (
        f"<b>{METHOD_LABEL.get(method, method)}</b> leads on {label} {scope}, "
        f"averaging <b>{score:.2f}</b>."
    )


# ---------------------------------------------------------------------------
# charts
# ---------------------------------------------------------------------------


def bar_chart(labels: list[str], values: list[Optional[float]], colours: list[str]) -> go.Figure:
    """A compact bar chart on the dark surface.

    Values that are None are dropped rather than plotted as zero, so a metric
    with no valid ground truth leaves a gap instead of a bar at the floor.
    """
    kept = [(l, v, c) for l, v, c in zip(labels, values, colours) if v is not None]
    figure = go.Figure()
    if kept:
        figure.add_trace(
            go.Bar(
                x=[l for l, _, _ in kept],
                y=[v for _, v, _ in kept],
                marker=dict(color=[c for _, _, c in kept], line=dict(width=0)),
                width=0.42,
                text=[f"{v:.2f}" for _, v, _ in kept],
                textposition="outside",
                textfont=dict(color=MUTED, size=11),
                hovertemplate="%{x}: %{y:.3f}<extra></extra>",
            )
        )
    figure.update_layout(
        height=252,
        margin=dict(l=6, r=6, t=26, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.45,
        yaxis=dict(
            range=[0, 1.06], gridcolor=GRID, zerolinecolor=GRID,
            tickfont=dict(color=MUTED, size=10), tickformat=".2f",
        ),
        xaxis=dict(
            showgrid=False, tickfont=dict(color="#B9C1D4", size=11),
            linecolor=GRID,
        ),
    )
    return figure


def method_bar_chart(
    scores: Dict[str, Optional[float]],
    comparisons: pd.DataFrame,
    metric_label: str,
) -> go.Figure:
    """One bar per strategy, with significance brackets above.

    Brackets are drawn only for pairs that survived the Holm correction. An
    absent bracket is itself information: it means the two bars are within
    sampling noise of each other, however different they look.
    """
    kept = [(m, v) for m, v in scores.items() if v is not None]
    figure = go.Figure()
    if not kept:
        return figure

    order = [m for m in METHOD_ORDER if m in dict(kept)]
    values = [scores[m] for m in order]
    figure.add_trace(
        go.Bar(
            x=[METHOD_LABEL[m] for m in order],
            y=values,
            marker=dict(color=[BLUE, "#4A79DC", BLUE_DEEP][: len(order)], line=dict(width=0)),
            width=0.42,
            text=[f"{v:.3f}" for v in values],
            textposition="inside",
            textfont=dict(color="#EAF0FF", size=11),
            hovertemplate="%{x}: %{y:.4f}<extra></extra>",
        )
    )

    significant = comparisons[comparisons["significant"]] if not comparisons.empty else pd.DataFrame()
    top = max(values)
    step = 0.085
    level = 0
    position = {METHOD_LABEL[m]: i for i, m in enumerate(order)}
    for row in significant.to_dict("records"):
        a, b = METHOD_LABEL.get(row["method_a"]), METHOD_LABEL.get(row["method_b"])
        if a not in position or b not in position:
            continue
        x0, x1 = sorted((position[a], position[b]))
        y = top + step * (level + 1)
        figure.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y,
                         line=dict(color=GOLD, width=1.1))
        for x in (x0, x1):
            figure.add_shape(type="line", x0=x, x1=x, y0=y - 0.018, y1=y,
                             line=dict(color=GOLD, width=1.1))
        figure.add_annotation(
            x=(x0 + x1) / 2, y=y + 0.004,
            text=f"{stars(row['p'])}  p={row['p']:.1e}",
            showarrow=False, yanchor="bottom",
            font=dict(color=GOLD_SOFT, size=9.5),
        )
        level += 1

    headroom = top + step * (level + 1) + 0.07
    figure.update_layout(
        height=300,
        margin=dict(l=6, r=6, t=14, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        bargap=0.5,
        yaxis=dict(range=[0, max(1.02, headroom)], gridcolor=GRID, zerolinecolor=GRID,
                   tickfont=dict(color=MUTED, size=10), tickformat=".2f",
                   title=dict(text=metric_label, font=dict(color=MUTED, size=10))),
        xaxis=dict(showgrid=False, tickfont=dict(color="#B9C1D4", size=11), linecolor=GRID),
    )
    return figure


def significance_matrix(frame: pd.DataFrame) -> str:
    """Compact per-dataset grid: which strategy wins each metric, and how surely."""
    if frame.empty:
        return '<div class="rb-empty">No comparisons available.</div>'

    metrics = ["recall@10", "mrr", "ndcg@10"]
    pairs = sorted({(r["method_a"], r["method_b"]) for r in frame.to_dict("records")})
    head = "<tr><th>Comparison</th>" + "".join(f"<th>{m}</th>" for m in metrics) + "</tr>"

    body = []
    for a, b in pairs:
        cells = []
        for metric in metrics:
            match = frame[
                (frame["method_a"] == a) & (frame["method_b"] == b) & (frame["metric"] == metric)
            ]
            if match.empty:
                cells.append('<td class="rb-null-cell">—</td>')
                continue
            row = match.iloc[0]
            if not row["significant"]:
                cells.append(
                    '<td><span style="color:#5A6275;">n.s.</span>'
                    f'<span style="color:#4A5064;font-size:.7rem;"> p={row["p"]:.2f}</span></td>'
                )
            else:
                colour = GOLD
                cells.append(
                    f'<td><span style="color:{colour};font-weight:600;">'
                    f'{METHOD_LABEL.get(row["winner"], row["winner"])}</span>'
                    f'<span style="color:{MUTED};font-size:.7rem;"> {stars(row["p"])} '
                    f'{row["diff"]:+.3f}</span></td>'
                )
        label = f'{METHOD_LABEL.get(a, a)} vs {METHOD_LABEL.get(b, b)}'
        body.append(f"<tr><td>{label}</td>{''.join(cells)}</tr>")
    return f'<table class="rb-table">{head}{"".join(body)}</table>'


def significance_detail(frame: pd.DataFrame) -> str:
    """Full statistics behind every comparison, including the effect size."""
    if frame.empty:
        return '<div class="rb-empty">No comparisons available.</div>'
    head = (
        "<tr><th>Dataset</th><th>Metric</th><th>Comparison</th><th>n</th>"
        "<th>Difference</th><th>95% CI</th><th>p (Holm)</th><th>Verdict</th></tr>"
    )
    body = []
    for row in frame.to_dict("records"):
        domain = DATASET_DOMAIN.get(row["dataset"], "unknown")
        colour = DOMAIN_COLOURS.get(domain, MUTED)
        badge = (
            f'<span class="rb-badge" style="background:{colour}1F;color:{colour};">'
            f'<span class="rb-dot" style="background:{colour};"></span>{row["dataset"]}</span>'
        )
        if row["significant"]:
            verdict = (
                f'<span style="color:{GOLD};font-weight:600;">'
                f'{METHOD_LABEL.get(row["winner"], row["winner"])}</span> '
                f'<span style="color:{MUTED};">{stars(row["p"])}</span>'
            )
        else:
            verdict = '<span class="rb-null-cell">not significant</span>'
        spans_zero = row["ci_low"] <= 0 <= row["ci_high"]
        ci_style = ' style="color:#5A6275;"' if spans_zero else ""
        body.append(
            f'<tr><td>{badge}</td><td>{row["metric"]}</td>'
            f'<td>{METHOD_LABEL.get(row["method_a"])} vs {METHOD_LABEL.get(row["method_b"])}</td>'
            f'<td class="rb-num">{row["n"]}</td>'
            f'<td class="rb-num">{row["diff"]:+.4f}</td>'
            f'<td class="rb-num"{ci_style}>[{row["ci_low"]:+.3f}, {row["ci_high"]:+.3f}]</td>'
            f'<td class="rb-num">{row["p"]:.2e}</td><td>{verdict}</td></tr>'
        )
    return f'<table class="rb-table">{head}{"".join(body)}</table>'


def card_open(title: str, note: str) -> str:
    return (
        f'<div class="rb-card"><div class="rb-card-head">'
        f'<span class="rb-card-title">{title}</span>'
        f'<span class="rb-card-note">{note}</span></div>'
    )


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def fmt(value: Optional[float], good_low=False) -> str:
    """Render a metric, or an em dash when there is no valid value.

    An em dash is the visual counterpart of SQL NULL: it says the measurement
    does not exist, which is a different statement from a score of zero.
    """
    if value is None or pd.isna(value):
        return '<span class="rb-null-cell">—</span>'
    if good_low is not None and good_low:
        css = "rb-good" if value <= 0.10 else ("rb-bad" if value >= 0.20 else "")
        return f'<span class="rb-num {css}">{value:.2f}</span>'
    return f'<span class="rb-num">{value:.2f}</span>'


def results_table(retrieval: pd.DataFrame, generation: pd.DataFrame, k: int) -> str:
    merged = retrieval.merge(
        generation[["dataset", "method", "faithfulness", "hallucination"]],
        on=["dataset", "method"], how="left",
    ) if not generation.empty else retrieval.assign(faithfulness=None, hallucination=None)

    merged["__d"] = merged["domain"].map({d: i for i, d in enumerate(DOMAIN_ORDER)}).fillna(9)
    merged["__m"] = merged["method"].map({m: i for i, m in enumerate(METHOD_ORDER)}).fillna(9)
    merged = merged.sort_values(["__d", "dataset", "__m"])

    head = (
        "<tr><th>Domain</th><th>Dataset</th><th>Strategy</th>"
        f"<th>Recall@{k}</th><th>MRR</th><th>NDCG</th>"
        "<th>Faithfulness</th><th>Hallucination</th></tr>"
    )
    body = []
    for row in merged.to_dict("records"):
        colour = DOMAIN_COLOURS.get(row["domain"], MUTED)
        badge = (
            f'<span class="rb-badge" style="background:{colour}1F;color:{colour};">'
            f'<span class="rb-dot" style="background:{colour};"></span>{row["domain"]}</span>'
        )
        tag = f'<span class="rb-tag">{METHOD_LABEL.get(row["method"], row["method"])}</span>'
        body.append(
            f"<tr><td>{badge}</td><td>{row['dataset']}</td><td>{tag}</td>"
            f"<td>{fmt(row.get('recall'))}</td>"
            f"<td>{fmt(row.get('mrr'))}</td>"
            f"<td>{fmt(row.get('ndcg'))}</td>"
            f"<td>{fmt(row.get('faithfulness'))}</td>"
            f"<td>{fmt(row.get('hallucination'), good_low=True)}</td></tr>"
        )
    return f'<table class="rb-table">{head}{"".join(body)}</table>'


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="RAG Benchmark Dashboard", layout="wide", page_icon="◆")
    st.markdown(CSS, unsafe_allow_html=True)

    config = load_config()
    db_path = config.paths.database_path
    if not db_path.exists():
        st.markdown(
            '<div class="rb-empty" style="margin-top:3rem;">'
            f"<b>No results database yet.</b><br>Expected at <code>{db_path}</code>.<br><br>"
            "Build one with:<br>"
            "<code>python -m cli build-corpus --all</code><br>"
            "<code>python -m cli build-indexes --all</code><br>"
            "<code>python -m cli benchmark --all</code></div>",
            unsafe_allow_html=True,
        )
        return

    aggregates, generations = read_tables(str(db_path), db_path.stat().st_mtime)
    ks = available_ks(aggregates)
    k = 10 if 10 in ks else ks[-1]
    retrieval_all = expand_retrieval(aggregates, k)
    generation_all = generation_summary(generations)

    # ---------------- sidebar ----------------
    with st.sidebar:
        if st.session_state.pop("__reset", False):
            for key in ("domain", "strategy", "focus"):
                st.session_state.pop(key, None)

        st.markdown('<div class="rb-filter-label">Domain</div>', unsafe_allow_html=True)
        domains = [d for d in DOMAIN_ORDER if d in set(retrieval_all.get("domain", []))]
        domain = st.selectbox(
            "domain", ["All domains"] + [d.capitalize() for d in domains],
            key="domain", label_visibility="collapsed",
        )

        st.markdown('<div class="rb-filter-label">Retrieval strategy</div>', unsafe_allow_html=True)
        strategy = st.selectbox(
            "strategy", ["All strategies"] + [METHOD_LABEL[m] for m in METHOD_ORDER],
            key="strategy", label_visibility="collapsed",
        )

        st.markdown('<div class="rb-filter-label">Metric focus</div>', unsafe_allow_html=True)
        focus = st.selectbox(
            "focus", ["All metrics", "Recall@K", "MRR", "NDCG"],
            key="focus", label_visibility="collapsed",
        )

        st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)
        if st.button("↺  Reset filters"):
            st.session_state["__reset"] = True
            st.rerun()

    # ---------------- filtering ----------------
    retrieval = retrieval_all.copy()
    generation = generation_all.copy()
    if domain != "All domains":
        retrieval = retrieval[retrieval["domain"] == domain.lower()]
        if not generation.empty:
            generation = generation[generation["domain"] == domain.lower()]
    if strategy != "All strategies":
        method = {v: kk for kk, v in METHOD_LABEL.items()}[strategy]
        retrieval = retrieval[retrieval["method"] == method]
        if not generation.empty:
            generation = generation[generation["method"] == method]

    # ---------------- header ----------------
    n_runs = int(aggregates["run_id"].nunique()) if not aggregates.empty else 0
    top = leading_strategy(retrieval[retrieval["recall"].notna()], "recall", "n_scoreable")
    top_label = METHOD_LABEL.get(top[0], "—") if top else "—"
    best_h = None
    if not generation.empty and generation["hallucination"].notna().any():
        best_h = float(generation["hallucination"].min())
    h_label = f"{best_h * 100:.0f}%" if best_h is not None else "—"

    st.markdown(
        f"""<div class="rb-header">
          <div>
            <div class="rb-eyebrow">Evaluation suite</div>
            <div class="rb-title">RAG Benchmark <em>Dashboard</em></div>
            <div class="rb-sub">legal · medical · scientific &nbsp;/&nbsp;
              3 retrieval strategies × {len(domains)} domains</div>
          </div>
          <div class="rb-stats">
            <div class="rb-stat"><div class="rb-stat-value">{n_runs}</div>
              <div class="rb-stat-label">Runs loaded</div></div>
            <div class="rb-stat"><div class="rb-stat-value">{top_label}</div>
              <div class="rb-stat-label">Top strategy</div></div>
            <div class="rb-stat"><div class="rb-stat-value" style="color:{GOLD}">{h_label}</div>
              <div class="rb-stat-label">Best hallucination</div></div>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ---------------- insight (sidebar, needs filtered data) ----------------
    with st.sidebar:
        shown, total = len(retrieval), len(retrieval_all)
        st.markdown(
            f"""<div class="rb-insight">
              <div class="rb-insight-label">Smart insight</div>
              <div class="rb-insight-body">
                Showing <b>{shown}</b> of {total} rows.<br>
                {build_insight(retrieval, generation, focus)}
              </div></div>""",
            unsafe_allow_html=True,
        )

    # ---------------- significance ----------------
    sig_all = load_significance(str(db_path), db_path.stat().st_mtime)
    sig = sig_all.copy()
    if not sig.empty:
        sig["domain"] = sig["dataset"].map(DATASET_DOMAIN).fillna("unknown")
        if domain != "All domains":
            sig = sig[sig["domain"] == domain.lower()]

    tab_overview, tab_significance = st.tabs(["  Overview  ", "  Significance  "])

    # ================= OVERVIEW =================
    with tab_overview:
        left, right = st.columns(2, gap="medium")

        with left:
            metric_key = METRIC_KEY.get(focus)
            datasets_shown = retrieval["dataset"].unique() if not retrieval.empty else []
            # Significance is a statement about one dataset and one metric, so
            # brackets are only meaningful once the filters isolate that pair.
            per_method_view = len(datasets_shown) == 1 and metric_key is not None

            st.markdown(card_open("Retrieval scores", "0–1 scale"), unsafe_allow_html=True)
            if per_method_view:
                only = datasets_shown[0]
                column = {"recall@10": "recall", "mrr": "mrr", "ndcg@10": "ndcg"}[metric_key]
                scores = {}
                for method in METHOD_ORDER:
                    block = retrieval[retrieval["method"] == method]
                    value = block[column].iloc[0] if not block.empty else None
                    scores[method] = None if value is None or pd.isna(value) else float(value)
                pairs = (
                    sig[(sig["dataset"] == only) & (sig["metric"] == metric_key)]
                    if not sig.empty else pd.DataFrame()
                )
                if any(v is not None for v in scores.values()):
                    st.plotly_chart(
                        method_bar_chart(scores, pairs, f"{only} · {focus}"),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                    legend = "Brackets mark Holm-corrected significant pairs"
                else:
                    st.markdown(
                        '<div class="rb-empty">No scoreable retrieval metrics '
                        "under these filters.</div>", unsafe_allow_html=True,
                    )
                    legend = "No scoreable metrics"
            else:
                values = [
                    weighted_mean(retrieval, "recall", "n_scoreable") if focus in ("All metrics", "Recall@K") else None,
                    weighted_mean(retrieval, "mrr", "n_scoreable") if focus in ("All metrics", "MRR") else None,
                    weighted_mean(retrieval, "ndcg", "n_scoreable") if focus in ("All metrics", "NDCG") else None,
                ]
                if any(v is not None for v in values):
                    st.plotly_chart(
                        bar_chart([f"Recall@{k}", "MRR", "NDCG"], values, RETRIEVAL_COLOURS),
                        use_container_width=True, config={"displayModeBar": False},
                    )
                    legend = "Weighted by scoreable queries · pick one dataset and metric for significance"
                else:
                    st.markdown(
                        '<div class="rb-empty">No scoreable retrieval metrics '
                        "under these filters.</div>", unsafe_allow_html=True,
                    )
                    legend = "No scoreable metrics"
            st.markdown(
                f'<div class="rb-legend"><span class="rb-dot" style="background:{BLUE}"></span>'
                f"{legend}</div></div>",
                unsafe_allow_html=True,
            )

        with right:
            g_values = [
                simple_mean(generation["faithfulness"]) if not generation.empty else None,
                simple_mean(generation["hallucination"]) if not generation.empty else None,
            ]
            st.markdown(card_open("Generation scores", "0–1 scale"), unsafe_allow_html=True)
            if any(v is not None for v in g_values):
                st.plotly_chart(
                    bar_chart(["Faithfulness", "Hallucination"], g_values, GENERATION_COLOURS),
                    use_container_width=True, config={"displayModeBar": False},
                )
            else:
                st.markdown(
                    '<div class="rb-empty">'
                    "<b>No scored answers yet.</b><br>"
                    "Retrieval is complete and unaffected.<br><br>"
                    "Generate and score with:<br>"
                    "<code>python -m cli generate-all --all</code><br>"
                    "<code>python -m cli score-answers</code></div>",
                    unsafe_allow_html=True,
                )
            st.markdown(
                f'<div class="rb-legend"><span class="rb-dot" style="background:{GOLD}"></span>'
                "Faithfulness &amp; hallucination rate</div></div>",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        st.markdown(
            card_open("Raw results", f"{len(retrieval)} rows")
            + results_table(retrieval, generation, k) + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="color:{MUTED};font-size:.72rem;margin-top:.7rem;">'
            "— marks a metric with no valid ground truth, which is distinct from a "
            "score of 0.00 meaning nothing relevant was retrieved.</div>",
            unsafe_allow_html=True,
        )

    # ================= SIGNIFICANCE =================
    with tab_significance:
        if sig.empty:
            st.markdown(
                '<div class="rb-empty"><b>No comparisons available.</b><br>'
                "Significance testing needs per-query scores from a completed "
                "benchmark run.<br><br><code>python -m cli benchmark --all</code></div>",
                unsafe_allow_html=True,
            )
        else:
            total = len(sig)
            confirmed = int(sig["significant"].sum())
            st.markdown(
                f'<div class="rb-insight" style="margin:.2rem 0 1.1rem;">'
                f'<div class="rb-insight-label">What this tab answers</div>'
                f'<div class="rb-insight-body">'
                f"<b>{confirmed} of {total}</b> pairwise comparisons survive Holm correction "
                f"at α=0.05. The rest are gaps you cannot claim: whatever the bars show, "
                f"those differences are within sampling noise. "
                f"Paired Wilcoxon signed-rank on per-query scores, bootstrap intervals "
                f"for effect size.</div></div>",
                unsafe_allow_html=True,
            )

            for name in sorted(sig["dataset"].unique()):
                block = sig[sig["dataset"] == name]
                won = int(block["significant"].sum())
                st.markdown(
                    card_open(name, f"{won}/{len(block)} significant")
                    + significance_matrix(block) + "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<div style='height:.7rem'></div>", unsafe_allow_html=True)

            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            st.markdown(
                card_open("All comparisons", f"{total} tests")
                + significance_detail(sig) + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="color:{MUTED};font-size:.72rem;margin-top:.7rem;line-height:1.6;">'
                "*** p&lt;0.001 &nbsp;·&nbsp; ** p&lt;0.01 &nbsp;·&nbsp; * p&lt;0.05, "
                "all Holm-corrected within each dataset and metric.<br>"
                "A greyed interval spans zero, which is the same statement as "
                "&ldquo;not significant&rdquo; read as an effect size. "
                "Datasets with no retrieval ground truth are absent entirely — "
                "they are not tested against zeros.</div>",
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
