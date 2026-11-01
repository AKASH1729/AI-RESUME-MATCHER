"""
Central configuration for the AI Job Scraper & Resume Matcher project.
Keep model names / paths here so they're easy to swap out.
"""

from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_FILE = DATA_DIR / "jobs.json"
INDEX_FILE = DATA_DIR / "job_index.faiss"
INDEX_META_FILE = DATA_DIR / "job_index_meta.json"

# ---------- Scraper ----------
# Public JSON job-board API (no auth/key needed). Swap this out or add
# more sources in scraper.py if you want to pull from other boards.
ARBEITNOW_API = "https://www.arbeitnow.com/api/job-board-api"
SCRAPE_MAX_PAGES = 3          # how many pages of results to pull
REQUEST_TIMEOUT = 15          # seconds
REQUEST_DELAY = 1.0           # polite delay between requests (seconds)

# ---------- Embeddings (RAG retrieval) ----------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RETRIEVAL = 5           # how many jobs to retrieve per resume query

# ---------- LLM (Hugging Face transformers) ----------
# Small instruction-tuned model that runs reasonably on CPU.
# Swap for a bigger model (e.g. "microsoft/Phi-3-mini-4k-instruct")
# if you have a GPU available.
LLM_MODEL = "google/flan-t5-base"
LLM_MAX_NEW_TOKENS = 200

# ---------- Matching ----------
MATCH_SCORE_MIN = 0
MATCH_SCORE_MAX = 100
