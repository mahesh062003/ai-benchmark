"""NLI hallucination detection: what fraction of the answer is unsupported?

A supervised NLI cross-encoder scores each answer sentence against each
retrieved chunk as premise. A sentence is *supported* when at least one chunk
entails it above ``entail_threshold``. It is *contradicted* only when some
chunk predicts contradiction as its winning class above
``contradict_threshold`` -- not merely when contradiction outscores a
negligible entailment, which would label flatly neutral sentences as
contradictions. Everything else is *neutral*.

  hallucination = unsupported sentences / total sentences, in [0, 1]

Higher is worse, the opposite direction to faithfulness. "Unsupported" means
the context does not entail the sentence; that includes both outright
contradiction and content the context is merely silent about. Both components
are kept in ``hallucination_json`` so they can be separated after the fact.

Known limitations, recorded rather than hidden:

* The cross-encoder truncates a premise+hypothesis pair at 512 tokens, so a
  sentence supported only by the tail of a long chunk can be scored
  unsupported. Chunks are 256 tokens under the default configuration, which
  keeps this rare but not impossible.
* Sentence splitting is regex-based, so an abbreviation can split a sentence.

The scorer returns ``None`` rather than a number whenever it cannot produce a
defensible score. ``None`` is written as SQL NULL, never as 0.0 -- a failed
measurement is not a zero score.

The companion measure is :mod:`evaluation.faithfulness`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from core.logging_setup import get_logger
from core.settings import GenerationConfig
from evaluation._shared import DEFAULT_NLI_MODEL, split_sentences

log = get_logger("hallucination")


@dataclass
class HallucinationResult:
    score: Optional[float]
    model: str
    sentences: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def contradiction_rate(self) -> Optional[float]:
        if not self.sentences:
            return None
        return sum(s["label"] == "contradiction" for s in self.sentences) / len(self.sentences)

    def to_json(self) -> str:
        return json.dumps(
            {
                "nli_model": self.model, "sentences": self.sentences,
                "n_sentences": len(self.sentences),
                "n_unsupported": sum(s["label"] != "entailment" for s in self.sentences),
                "n_contradicted": sum(s["label"] == "contradiction" for s in self.sentences),
                "contradiction_rate": self.contradiction_rate,
                "error": self.error,
            },
            ensure_ascii=False,
        )


class HallucinationScorer:
    """Sentence-level NLI grounding check with a cross-encoder."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        entail_threshold: float = 0.5,
        contradict_threshold: float = 0.5,
        batch_size: int = 32,
        max_length: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.entail_threshold = entail_threshold
        self.contradict_threshold = contradict_threshold
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._model = None
        self._label_index: Dict[str, int] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("loading NLI cross-encoder %s", self.model_name)
            self._model = CrossEncoder(
                self.model_name, max_length=self.max_length, device=self.device
            )
            # Label order differs between NLI checkpoints; read it off the
            # config rather than assuming the common (contradiction, entailment,
            # neutral) ordering.
            id2label = getattr(self._model.config, "id2label", None) or {}
            self._label_index = {
                str(name).lower(): int(index) for index, name in id2label.items()
            }
            missing = {"entailment", "contradiction"} - set(self._label_index)
            if missing:
                raise ValueError(
                    f"{self.model_name} does not expose NLI labels {sorted(missing)}; "
                    f"found {sorted(self._label_index)}"
                )
        return self._model

    def score(self, answer: str, context: Sequence[str]) -> HallucinationResult:
        if not answer or not answer.strip():
            return HallucinationResult(None, self.model_name, error="empty answer")
        if not context:
            return HallucinationResult(None, self.model_name, error="no context retrieved")

        sentences = split_sentences(answer)
        if not sentences:
            return HallucinationResult(
                None, self.model_name, error="answer has no scoreable sentences"
            )
        return self._score_sentences(sentences, list(context))

    def score_batch(self, items: Sequence[tuple]) -> List[HallucinationResult]:
        """Score many (answer, context) pairs, batching every NLI pair together.

        One padded forward pass over all pairs is far faster on GPU than one
        call per answer, which matters at benchmark scale.
        """
        prepared = []
        results: List[Optional[HallucinationResult]] = []
        pairs: List[tuple] = []

        for answer, context in items:
            if not answer or not answer.strip():
                results.append(HallucinationResult(None, self.model_name, error="empty answer"))
                prepared.append(None)
                continue
            context = list(context)
            if not context:
                results.append(
                    HallucinationResult(None, self.model_name, error="no context retrieved")
                )
                prepared.append(None)
                continue
            sentences = split_sentences(answer)
            if not sentences:
                results.append(
                    HallucinationResult(
                        None, self.model_name, error="answer has no scoreable sentences"
                    )
                )
                prepared.append(None)
                continue
            start = len(pairs)
            for sentence in sentences:
                for chunk in context:
                    pairs.append((chunk, sentence))
            prepared.append((sentences, context, start))
            results.append(None)

        if not pairs:
            return [r for r in results if r is not None] or []

        logits = self._predict(pairs)
        for position, entry in enumerate(prepared):
            if entry is None:
                continue
            sentences, context, start = entry
            width = len(context)
            block = [
                logits[start + i * width : start + (i + 1) * width]
                for i in range(len(sentences))
            ]
            results[position] = self._assemble(sentences, block)
        return [r for r in results if r is not None]

    def _score_sentences(self, sentences, context) -> HallucinationResult:
        pairs = [(chunk, sentence) for sentence in sentences for chunk in context]
        logits = self._predict(pairs)
        width = len(context)
        block = [logits[i * width : (i + 1) * width] for i in range(len(sentences))]
        return self._assemble(sentences, block)

    def _predict(self, pairs):
        import numpy as np

        raw = self.model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False,
            convert_to_numpy=True, apply_softmax=False,
        )
        raw = np.asarray(raw, dtype="float64")
        shifted = raw - raw.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def _assemble(self, sentences, block) -> HallucinationResult:
        entail_index = self._label_index["entailment"]
        contra_index = self._label_index["contradiction"]

        detail: List[Dict] = []
        for sentence, probabilities in zip(sentences, block):
            entailments = probabilities[:, entail_index]
            contradictions = probabilities[:, contra_index]
            entail = float(entailments.max())
            best_chunk = int(entailments.argmax())

            # Contradiction is only claimed when some chunk actually predicts
            # it as the winning class, not merely when it edges out a
            # negligible entailment. Comparing the two maxima alone labels
            # ent=0.001 / con=0.003 -- a flatly neutral pair -- as a
            # contradiction, which would wildly overstate how often models
            # assert something the context denies.
            contradicting = int(contradictions.argmax())
            contradict = float(contradictions[contradicting])
            dominant = int(probabilities[contradicting].argmax()) == contra_index

            if entail >= self.entail_threshold:
                label = "entailment"
            elif dominant and contradict >= self.contradict_threshold:
                label = "contradiction"
            else:
                label = "neutral"
            detail.append(
                {
                    "sentence": sentence[:300], "label": label,
                    "entailment": round(entail, 4),
                    "contradiction": round(contradict, 4),
                    "best_chunk": best_chunk,
                }
            )
        unsupported = sum(d["label"] != "entailment" for d in detail)
        return HallucinationResult(
            score=unsupported / len(detail), model=self.model_name, sentences=detail
        )
