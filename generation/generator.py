"""Optional generation stage using a local Ollama model.

Nothing here is imported by the retrieval path, and no Ollama connection is
opened unless generation is explicitly enabled, so the retrieval benchmark runs
on a machine with no Ollama installed.

Implemented answer-quality measures
-----------------------------------
``choice_correct``  For the multiple-choice datasets -- MedQA, SciQ and
                    CaseHOLD, each of which supplies ``options`` and
                    ``answer_idx`` in its query metadata -- the generated
                    answer is matched to an option and compared with the gold
                    option. This is a real, checkable measure that needs no
                    retrieval ground truth, which is what makes MedQA
                    evaluable. A dataset supplying neither leaves the column
                    NULL: nothing is scored wrong merely because it was never
                    asked as a choice.
``exact_match``     Normalised string equality against the reference answer,
                    for datasets with a short free-text answer (PubMedQA's
                    yes/no/maybe decision).

Scored separately
-----------------
``faithfulness`` and ``hallucination`` are **not** computed here. Both need a
model of their own -- an LLM judge and an NLI cross-encoder respectively -- and
live in ``evaluation.faithfulness`` and ``evaluation.hallucination``, applied as a second pass over stored
answers by ``ragbench score-answers``. Keeping them out of the generation loop
means a scorer can be re-run, re-thresholded or replaced without re-generating
several thousand answers. Until that pass runs, both columns are SQL NULL.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from core.settings import Config, GenerationConfig
from core.logging_setup import get_logger
from core.types import Chunk, Query

log = get_logger("generation")

SYSTEM_PROMPT = (
    "You are a careful domain expert. Answer the question using only the "
    "provided context. If the context does not contain the answer, say so "
    "explicitly rather than guessing."
)


def build_prompt(query: Query, chunks: Sequence[Chunk], max_chunks: int = 5) -> str:
    """Assemble the RAG prompt from the query and its retrieved context."""
    selected = list(chunks)[:max_chunks]
    context = "\n\n".join(
        f"[{i}] {chunk.text.strip()}" for i, chunk in enumerate(selected, start=1)
    )
    options = query.metadata.get("options")
    parts = [SYSTEM_PROMPT, "", "Context:", context or "(no context retrieved)", ""]
    if options:
        rendered = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
        parts += [
            f"Question: {query.text}", "", "Options:", rendered, "",
            "Answer with the single letter of the correct option, then a brief "
            "justification grounded in the context.",
        ]
    else:
        parts += [f"Question: {query.text}", "", "Answer:"]
    return "\n".join(parts)


class OllamaClient:
    """Minimal client for a local Ollama server."""

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.config.host)
        return self._client

    def available(self) -> bool:
        """True if the server responds and the configured model is present."""
        try:
            listed = self.client.list()
        except Exception as exc:
            log.warning("Ollama not reachable at %s: %s", self.config.host, exc)
            return False

        # ollama-python returns a dict on older versions and a pydantic
        # ListResponse on >=0.4; support both rather than pinning a version.
        entries = listed.get("models", []) if isinstance(listed, dict) else getattr(
            listed, "models", []
        )
        names = []
        for model in entries or []:
            if isinstance(model, dict):
                name = model.get("model") or model.get("name")
            else:
                name = getattr(model, "model", None) or getattr(model, "name", None)
            if name:
                names.append(str(name))

        wanted = self.config.model
        if not any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names):
            log.warning(
                "Ollama model %r not found. Available: %s. Pull it with: ollama pull %s",
                wanted, ", ".join(names) or "(none)", wanted,
            )
            return False
        return True

    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.config.model,
            prompt=prompt,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
        )
        if isinstance(response, dict):
            return response.get("response", "")
        return getattr(response, "response", "") or ""


# --------------------------------------------------------------------------
# answer-quality measures
# --------------------------------------------------------------------------

_NORMALISE = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    return _NORMALISE.sub(" ", (text or "").lower()).strip()


def exact_match(prediction: str, reference: Optional[str]) -> Optional[float]:
    """Normalised equality, or the reference appearing as whole words.

    Containment is matched on word boundaries, never as a raw substring. A raw
    substring test silently scores PubMedQA's ``no`` as correct inside "does
    **no**t provide an answer", turning refusals into perfect scores and
    inflating every model's accuracy on the yes/no/maybe datasets.

    NULL when the dataset supplies no reference answer.
    """
    if reference is None or not str(reference).strip():
        return None
    predicted = normalise(prediction)
    expected = normalise(str(reference))
    if not expected:
        return None
    if predicted == expected:
        return 1.0
    return 1.0 if re.search(rf"\b{re.escape(expected)}\b", predicted) else 0.0


def extract_choice(prediction: str, options: Dict[str, str]) -> Optional[str]:
    """Recover the chosen option key from free-form model output.

    Tries an explicit leading letter first, then falls back to matching option
    text. Returns None when nothing matches, which is scored as incorrect
    rather than silently skipped.
    """
    if not options:
        return None
    text = (prediction or "").strip()
    if not text:
        return None

    keys = sorted(options)
    letters = "".join(re.escape(k) for k in keys)
    patterns = [
        rf"^\s*\(?([{letters}])\)?[\.\):,\s]",
        rf"\banswer\s*(?:is)?\s*:?\s*\(?([{letters}])\)?\b",
        rf"^\s*([{letters}])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            found = match.group(1).upper()
            if found in options:
                return found

    normalised = normalise(text)
    for key, value in options.items():
        candidate = normalise(value)
        if candidate and candidate in normalised:
            return key
    return None


def choice_correct(
    prediction: str, options: Dict[str, str], gold_key: Optional[str]
) -> Optional[float]:
    """1.0/0.0 for multiple-choice datasets; NULL when not multiple choice."""
    if not options or not gold_key:
        return None
    return 1.0 if extract_choice(prediction, options) == gold_key else 0.0


@dataclass
class GenerationEvaluation:
    query_id: str
    prompt: str
    answer: Optional[str]
    reference_answer: Optional[str]
    context_chunk_ids: List[str]
    latency_ms: Optional[float]
    exact_match: Optional[float]
    choice_correct: Optional[float]
    faithfulness: Optional[float] = None   # NULL until `score-answers` runs
    hallucination: Optional[float] = None  # NULL until `score-answers` runs
    error: Optional[str] = None

    def to_record(self) -> dict:
        return {
            "query_id": self.query_id, "prompt": self.prompt, "answer": self.answer,
            "reference_answer": self.reference_answer,
            "context_chunk_ids": self.context_chunk_ids, "latency_ms": self.latency_ms,
            "exact_match": self.exact_match, "choice_correct": self.choice_correct,
            "faithfulness": self.faithfulness, "hallucination": self.hallucination,
            "error": self.error,
        }


def generate_for_query(
    client: OllamaClient,
    query: Query,
    chunks: Sequence[Chunk],
    config: Config,
) -> GenerationEvaluation:
    prompt = build_prompt(query, chunks, config.generation.max_context_chunks)
    context_ids = [c.chunk_id for c in list(chunks)[: config.generation.max_context_chunks]]
    started = time.time()
    try:
        answer = client.generate(prompt)
        error = None
    except Exception as exc:
        log.error("generation failed for %s: %s", query.query_id, exc)
        return GenerationEvaluation(
            query_id=query.query_id, prompt=prompt, answer=None,
            reference_answer=query.answer, context_chunk_ids=context_ids,
            latency_ms=None, exact_match=None, choice_correct=None, error=str(exc),
        )
    latency = (time.time() - started) * 1000.0

    options = query.metadata.get("options") or {}
    gold_key = query.metadata.get("answer_idx")
    return GenerationEvaluation(
        query_id=query.query_id, prompt=prompt, answer=answer,
        reference_answer=query.answer, context_chunk_ids=context_ids,
        latency_ms=latency,
        exact_match=exact_match(answer, query.answer),
        choice_correct=choice_correct(answer, options, gold_key),
        error=error,
    )
