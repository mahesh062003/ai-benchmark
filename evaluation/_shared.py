"""Helpers shared by the two answer-grounding measures.

Both scorers must survive the output of a small local model, which wraps JSON
in prose far more often than a hosted model does, and both work at sentence
granularity. Keeping these here means the two measures stay genuinely
independent instruments rather than one importing the other.
"""

from __future__ import annotations

import json
import re
from typing import List

DEFAULT_JUDGE_MODEL = "llama3.1"
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

# Sentence boundaries are terminal punctuation followed by a capital or digit.
# Abbreviations are masked before splitting so that "et al." and friends do not
# split one sentence into two.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "Dr.", "Fig.", "No.", "cf.")


def split_sentences(text: str, min_chars: int = 15) -> List[str]:
    """Split an answer into sentences for sentence-level NLI.

    Fragments shorter than ``min_chars`` are dropped: a bare "Yes." or an
    option letter carries no propositional content to verify, and counting it
    as an unsupported sentence would inflate the hallucination rate.
    """
    if not text or not text.strip():
        return []
    guarded = text
    for i, abbreviation in enumerate(_ABBREVIATIONS):
        guarded = guarded.replace(abbreviation, f"\x00{i}\x00")

    sentences = []
    # Split on line breaks first. Multiple-choice answers open with a bare
    # option letter on its own line ("D\n\nThe patient's presentation..."), and
    # without this the letter is glued to the first sentence and pollutes the
    # hypothesis handed to the NLI model.
    for block in re.split(r"\n+", guarded):
        for piece in _SENTENCE_SPLIT.split(block):
            for i, abbreviation in enumerate(_ABBREVIATIONS):
                piece = piece.replace(f"\x00{i}\x00", abbreviation)
            piece = piece.strip()
            if len(piece) >= min_chars:
                sentences.append(piece)
    return sentences


def extract_json(text: str):
    """Recover a JSON value from a local model's reply.

    Small local models wrap JSON in prose or code fences far more often than
    hosted models do, so the first parse attempt is on the raw text, then on a
    fenced block, then on the outermost bracketed span.
    """
    if not text:
        return None
    for candidate in _json_candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _json_candidates(text: str):
    yield text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        yield fenced.group(1).strip()
    for opening, closing in (("[", "]"), ("{", "}")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start != -1 and end > start:
            yield text[start : end + 1]
