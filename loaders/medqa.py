"""MedQA -- USMLE-style medical exam questions over a textbook corpus (medical).

Audited shape of ``datasets/Medical/MedQA``:
  ``questions/{train,dev,test}.jsonl`` hold 10,178 / 1,272 / 1,273 records with
  fields question, answer, options (5 choices), answer_idx, meta_info
  (step1 / step2&3). ``answer_idx`` and ``answer`` agree on every record, and
  the three splits share no question text, so the official split is clean and
  is preserved as-is.
  ``textbooks/`` holds 18 medical reference texts totalling 89.4 MB.

Ground truth: UNSUPPORTED. This is the most important negative finding of the
dataset audit, so it is spelled out.

  MedQA ships a genuine external retrieval corpus -- the 18 textbooks -- which
  makes it the only dataset here with a realistic open-domain medical corpus.
  What it does *not* ship is any annotation linking a question to the textbook
  passage that supports it. There are no evidence spans, no passage ids, no
  source references, not even a book-level attribution.

  Therefore Recall@k, MRR and nDCG@k cannot be computed for MedQA without
  fabricating relevance, and this framework reports them as NULL.

Rejected alternatives, recorded so the decision can be defended:
  * Marking a chunk relevant because it contains the answer string. The answer
    options are clinical sentences ("Tell the attending that he cannot fail to
    disclose this mistake"), so containment would almost never fire, and where
    it did it would measure string overlap rather than evidential support. This
    is the HEURISTIC class and is deliberately not used.
  * Treating the question's own text as its corpus, which would make retrieval
    trivially self-satisfying and is meaningless.
  * Recording unavailable metrics as 0.0, which would be read as "the retriever
    failed" when the truth is "there was nothing to score against".

What MedQA *is* used for: the end-to-end generation stage. Retrieval over the
textbook corpus supplies context, the generator answers the multiple-choice
question, and multiple-choice accuracy is a legitimate downstream measure that
requires no retrieval ground truth. Retrieved passages are still persisted so
they can be inspected qualitatively.
"""

from __future__ import annotations

import json
from typing import List

from core.settings import ChunkConfig
from core.ids import doc_id as make_doc_id
from core.ids import query_id as make_query_id
from core.logging_setup import SkipLog, get_logger
from core.types import (
    CorpusScope,
    DatasetSpec,
    Query,
    RelevanceClass,
    SourceDocument,
)
from .base import DatasetAdapter, LoadedSplit

log = get_logger("adapters.medqa")

_SPLITS = {"train": "train.jsonl", "dev": "dev.jsonl", "test": "test.jsonl"}


class MedQAAdapter(DatasetAdapter):
    spec = DatasetSpec(
        name="medqa",
        domain="medical",
        relevance_class=RelevanceClass.UNSUPPORTED,
        corpus_scope=CorpusScope.POOLED,
        splits=["train", "dev", "test"],
        query_unit="USMLE-style clinical vignette question",
        corpus_unit="fixed-size chunk of a medical textbook",
        relevance_source=(
            "NONE. MedQA provides no question-to-passage evidence annotation of "
            "any kind. Retrieval metrics are reported as NULL."
        ),
        limitations=(
            "No retrieval ground truth exists, so Recall@k / MRR / nDCG@k are "
            "unavailable (NULL, not zero) and MedQA contributes no rows to the "
            "retrieval comparison. Its role is the end-to-end generation "
            "evaluation, where multiple-choice accuracy is measurable without "
            "relevance judgements. The 89 MB textbook corpus is by far the "
            "largest here, so indexing cost dominates; restrict it with "
            "max_documents for smoke runs."
        ),
    )
    default_chunk = ChunkConfig(strategy="fixed", size_tokens=256, overlap_tokens=64)

    def load(self, split: str = "test") -> LoadedSplit:
        if split not in _SPLITS:
            raise ValueError(f"medqa split must be one of {sorted(_SPLITS)}; got {split!r}")
        questions_path = self._require(
            self.root / "Medical" / "MedQA" / "questions" / _SPLITS[split]
        )
        textbook_dir = self._require(self.root / "Medical" / "MedQA" / "textbooks")
        skips = SkipLog("medqa", split)
        ds_cfg = self.config.dataset(self.name)

        # --- corpus: the textbooks (shared across all splits) ---------------
        documents: List[SourceDocument] = []
        book_paths = sorted(textbook_dir.glob("*.txt"))
        if ds_cfg.max_documents:
            book_paths = book_paths[: ds_cfg.max_documents]
        for book_path in book_paths:
            text = book_path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                skips.skip("empty_textbook")
                continue
            name = book_path.stem
            documents.append(
                SourceDocument(
                    doc_id=make_doc_id("medqa", "textbooks", name),
                    dataset="medqa",
                    # Textbooks are split-independent; label them so it is
                    # obvious they are not drawn from the question split.
                    split="textbooks",
                    text=text,
                    title=name.replace("_", " "),
                    metadata={"textbook": name, "n_chars": len(text)},
                )
            )

        # --- queries --------------------------------------------------------
        queries: List[Query] = []
        with questions_path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    skips.skip("malformed_json_line")
                    continue
                question = str(record.get("question", "")).strip()
                if not question:
                    skips.skip("empty_question")
                    continue
                options = record.get("options") or {}
                answer_idx = record.get("answer_idx")
                if answer_idx not in options:
                    skips.skip("answer_idx_not_in_options")
                    continue

                queries.append(
                    Query(
                        query_id=make_query_id("medqa", split, f"{index:05d}"),
                        dataset="medqa",
                        split=split,
                        text=question,
                        scope_doc_id=None,
                        # Empty by design and by audit: no evidence exists.
                        evidence=[],
                        relevance_class=RelevanceClass.UNSUPPORTED,
                        answer=str(record.get("answer", "")),
                        metadata={
                            "row_index": index,
                            "options": {str(k): str(v) for k, v in options.items()},
                            "answer_idx": str(answer_idx),
                            "meta_info": str(record.get("meta_info", "")),
                            "answer_type": "multiple_choice",
                        },
                    )
                )

        if ds_cfg.max_queries:
            queries = queries[: ds_cfg.max_queries]

        stats = {
            "documents": len(documents),
            "queries": len(queries),
            "queries_with_evidence": 0,
            "textbook_chars": sum(len(d.text) for d in documents),
        }
        skips.report(log, kept=len(queries))
        log.info(
            "medqa/%s: %d textbooks (%.1f MB) as corpus, %d questions; "
            "retrieval ground truth UNSUPPORTED -> retrieval metrics will be NULL",
            split, len(documents), stats["textbook_chars"] / 1e6, len(queries),
        )
        return LoadedSplit(documents=documents, queries=queries, skips=skips, stats=stats)
