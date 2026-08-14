"""RAGAS faithfulness: is the answer supported by the context it was given?

The published RAGAS procedure, implemented directly against a local Ollama
judge rather than through the ``ragas`` package, which would pull a LangChain
and OpenAI-client dependency chain to reach the same two prompts:

  1. decompose the answer into atomic statements;
  2. for each statement, decide whether the retrieved context supports it;
  3. faithfulness = supported statements / total statements, in [0, 1].

The judge is recorded per row (``faithfulness_judge``). This matters because a
score is only meaningful relative to its rater, and because judging a model's
answer with that same model is self-evaluation -- by default the judge is one
fixed model for every row so that no model grades its own homework on a
different footing from its competitors.

An LLM judge is not a gold standard: this measures agreement with one judge
model, not ground truth. The scorer returns ``None`` rather than a number
whenever it cannot produce a defensible score (empty answer, no context, judge
failure, unparseable output). ``None`` is written as SQL NULL, never as 0.0 --
a failed measurement is not a zero score.

The companion measure is :mod:`evaluation.hallucination`, which asks the same
question with a supervised NLI model. They fail in different ways, which is the
point: agreement is evidence, disagreement is a flag for inspection.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from core.logging_setup import get_logger
from core.settings import GenerationConfig
from evaluation._shared import DEFAULT_JUDGE_MODEL, extract_json, split_sentences

log = get_logger("faithfulness")


STATEMENT_PROMPT = """\
You break answers into atomic factual statements.

Rewrite the ANSWER as a JSON array of short, self-contained factual statements.
Each statement must be understandable on its own: replace pronouns with the
thing they refer to. Do not add facts that are not in the answer. Do not
include opinions, hedges, or questions.

Question: {question}

ANSWER:
{answer}

Return ONLY a JSON array of strings, for example ["...", "..."].
"""

VERDICT_PROMPT = """\
You decide whether statements are supported by a context.

For each statement, decide whether it can be directly inferred from the
CONTEXT. Answer 1 if the context supports the statement, 0 if it does not or
if the context is silent about it. Judge only against the context; ignore
whether the statement is true in the world.

CONTEXT:
{context}

STATEMENTS:
{statements}

Return ONLY a JSON array with one object per statement, in the same order:
[{{"index": 1, "verdict": 1, "reason": "..."}}]
"""


@dataclass
class FaithfulnessResult:
    score: Optional[float]
    judge: str
    statements: List[str] = field(default_factory=list)
    verdicts: List[int] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "judge": self.judge, "statements": self.statements,
                "verdicts": self.verdicts, "reasons": self.reasons,
                "n_statements": len(self.statements),
                "n_supported": sum(self.verdicts) if self.verdicts else 0,
                "error": self.error,
            },
            ensure_ascii=False,
        )


class FaithfulnessScorer:
    """RAGAS faithfulness against a local Ollama judge."""

    def __init__(self, config: GenerationConfig, judge_model: str = DEFAULT_JUDGE_MODEL) -> None:
        from dataclasses import replace

        from generation.generator import OllamaClient

        self.judge_model = judge_model
        self.client = OllamaClient(
            replace(config, model=judge_model, enabled=True, temperature=0.0)
        )

    def available(self) -> bool:
        return self.client.available()

    def score(self, question: str, answer: str, context: Sequence[str]) -> FaithfulnessResult:
        if not answer or not answer.strip():
            return FaithfulnessResult(None, self.judge_model, error="empty answer")
        if not context:
            return FaithfulnessResult(None, self.judge_model, error="no context retrieved")

        try:
            statements = self._statements(question, answer)
        except Exception as exc:
            log.warning("statement decomposition failed: %s", exc)
            return FaithfulnessResult(None, self.judge_model, error=f"decompose: {exc}")

        if not statements:
            # No verifiable proposition (e.g. "I cannot answer from the
            # context"). That is not a faithfulness failure, so it is NULL.
            return FaithfulnessResult(
                None, self.judge_model, error="answer contains no factual statements"
            )

        try:
            verdicts, reasons = self._verdicts(context, statements)
        except Exception as exc:
            log.warning("statement verification failed: %s", exc)
            return FaithfulnessResult(
                None, self.judge_model, statements=statements, error=f"verify: {exc}"
            )

        if not verdicts:
            return FaithfulnessResult(
                None, self.judge_model, statements=statements,
                error="judge returned no parseable verdicts",
            )
        return FaithfulnessResult(
            score=sum(verdicts) / len(verdicts),
            judge=self.judge_model, statements=statements,
            verdicts=verdicts, reasons=reasons,
        )

    def _statements(self, question: str, answer: str) -> List[str]:
        reply = self.client.generate(
            STATEMENT_PROMPT.format(question=question.strip(), answer=answer.strip())
        )
        parsed = extract_json(reply)
        if isinstance(parsed, dict):
            for key in ("statements", "sentences", "facts"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            return []
        statements = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                statements.append(item.strip())
            elif isinstance(item, dict):
                value = item.get("statement") or item.get("text")
                if isinstance(value, str) and value.strip():
                    statements.append(value.strip())
        return statements[:20]

    def _verdicts(self, context: Sequence[str], statements: Sequence[str]):
        rendered_context = "\n\n".join(
            f"[{i}] {c.strip()}" for i, c in enumerate(context, start=1)
        )
        rendered_statements = "\n".join(
            f"{i}. {s}" for i, s in enumerate(statements, start=1)
        )
        reply = self.client.generate(
            VERDICT_PROMPT.format(
                context=rendered_context, statements=rendered_statements
            )
        )
        parsed = extract_json(reply)
        if isinstance(parsed, dict):
            for key in ("verdicts", "results", "answers"):
                if isinstance(parsed.get(key), list):
                    parsed = parsed[key]
                    break
        if not isinstance(parsed, list):
            return [], []

        verdicts: List[int] = []
        reasons: List[str] = []
        for item in parsed[: len(statements)]:
            if isinstance(item, dict):
                raw = item.get("verdict", item.get("supported"))
                reasons.append(str(item.get("reason", ""))[:300])
            else:
                raw = item
                reasons.append("")
            verdicts.append(_as_verdict(raw))
        return verdicts, reasons


def _as_verdict(raw) -> int:
    """Coerce a judge's verdict to 1/0, defaulting to 0 (unsupported).

    Defaulting an unreadable verdict to "unsupported" is the conservative
    direction: it can understate faithfulness but never invent support.
    """
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return 1 if raw >= 1 else 0
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in ("1", "yes", "true", "supported", "y"):
            return 1
    return 0
