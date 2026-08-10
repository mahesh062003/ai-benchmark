from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATASETS_DIR = PROJECT_ROOT / "datasets"

RESULTS_DIR = PROJECT_ROOT / "results"

CHUNKS_DIR = RESULTS_DIR / "chunks"

INDEX_DIR = RESULTS_DIR / "indexes"

DATABASE_PATH = RESULTS_DIR / "benchmark.db"


# ============================================================
# DEVELOPMENT SETTINGS
# ============================================================

DEV_MODE = True

SAMPLE_LIMIT = 500

TOP_K = 5

LLM_MODEL = "llama3.1:8b-instruct-q4_K_M"