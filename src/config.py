"""
Central configuration for the AI Benchmark project.

All paths, model names, and hyperparameters live here.
No other module should define configuration constants independently.
"""

from pathlib import Path

# ── Project layout ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # C:/AI BENCHMARK/
DATASETS_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR  = PROJECT_ROOT / "results"
CHUNKS_DIR   = RESULTS_DIR  / "chunks"
INDEX_DIR    = RESULTS_DIR  / "indexes"
DATABASE_PATH = RESULTS_DIR / "benchmark.db"

# ── Retrieval corpus and preprocessing ───────────────────────────────────────

CHUNK_SIZE    = 512   # characters (LangChain RecursiveCharacterTextSplitter)
CHUNK_OVERLAP = 50
TOP_K         = 5     # number of retrieved chunks per query

# ── Models ────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL       = "llama3.1:8b-instruct-q4_K_M"   # served by local Ollama

# ── Benchmark execution ───────────────────────────────────────────────────────

SAMPLE_LIMIT      = 500    # max query samples per dataset per run
RETRIEVAL_METHODS = ["BM25", "FAISS", "Hybrid"]

# ── Dataset file paths ────────────────────────────────────────────────────────
# Each entry gives the raw file path relative to PROJECT_ROOT.
# The corpus and query files can differ (to separate indexing from evaluation).

DATASET_PATHS = {
    "CUAD": {
        # Single combined split; paragraphs deduped at corpus-build time.
        "corpus": DATASETS_DIR / "Legal"    / "CUAD"    / "CUADv1.json",
        "queries": DATASETS_DIR / "Legal"   / "CUAD"    / "CUADv1.json",
    },
    "CaseHOLD": {
        # Test split for evaluation queries; train for corpus.
        "corpus":  DATASETS_DIR / "Legal"   / "LexGLUE" / "case_hold" / "case_hold_train.csv",
        "queries": DATASETS_DIR / "Legal"   / "LexGLUE" / "case_hold" / "case_hold_test.csv",
    },
    "MedQA": {
        # No external corpus; dev split used as query set.
        "corpus":  None,
        "queries": DATASETS_DIR / "Medical" / "MedQA"   / "questions" / "dev.jsonl",
    },
    "PubMedQA": {
        # Single labeled split; all abstracts indexed and queried.
        "corpus":  DATASETS_DIR / "Medical" / "PubMedQA" / "pqa_labeled" / "train_labeled.csv",
        "queries": DATASETS_DIR / "Medical" / "PubMedQA" / "pqa_labeled" / "train_labeled.csv",
    },
    "QASPER": {
        # Corpus = train + test papers; queries = test QA pairs.
        "corpus_train": DATASETS_DIR / "Scientific" / "qasper" / "train.parquet",
        "corpus_test":  DATASETS_DIR / "Scientific" / "qasper" / "test.parquet",
        "queries":      DATASETS_DIR / "Scientific" / "qasper" / "test.parquet",
    },
    "SciQ": {
        # Test split for both corpus and evaluation.
        "corpus":  DATASETS_DIR / "Scientific" / "SciQ" / "test.csv",
        "queries": DATASETS_DIR / "Scientific" / "SciQ" / "test.csv",
    },
}
