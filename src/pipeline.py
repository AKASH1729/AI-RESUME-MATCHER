"""
pipeline.py
-----------
Ties everything together end-to-end:

    resume file -> parsed text -> RAG retrieval (top-K relevant jobs)
    -> LLM scoring/explanation for each -> ranked results

Run directly for a CLI demo, or import `run_pipeline` from app.py.
"""

import logging
from typing import List

from config import TOP_K_RETRIEVAL, JOBS_FILE
from resume_parser import parse_resume
from scraper import load_jobs, fetch_from_api, save_jobs
from rag_engine import JobRAGIndex
from llm_matcher import LLMMatcher, MatchResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def ensure_jobs() -> list:
    """Load cached jobs, or scrape fresh ones if none exist yet."""
    jobs = load_jobs()
    if not jobs:
        log.info("No cached jobs found — scraping fresh listings...")
        jobs = fetch_from_api()
        if jobs:
            save_jobs(jobs)
    return jobs


def run_pipeline(resume_path: str, top_k: int = TOP_K_RETRIEVAL) -> List[MatchResult]:
    # 1. Parse resume
    resume_text = parse_resume(resume_path)
    log.info(f"Parsed resume ({len(resume_text)} chars).")

    # 2. Load / scrape jobs and build RAG index
    jobs = ensure_jobs()
    if not jobs:
        raise RuntimeError(
            "No job postings available and scraping failed (check network access). "
            f"You can also manually place a jobs list at {JOBS_FILE}."
        )

    rag = JobRAGIndex()
    rag.build(jobs)

    # 3. Retrieve top-K relevant postings
    retrieved = rag.retrieve(resume_text, top_k=top_k)
    log.info(f"Retrieved {len(retrieved)} candidate jobs via RAG.")

    # 4. LLM scores + explains each retrieved job
    matcher = LLMMatcher()
    results = [matcher.match(resume_text, job) for job, _sim in retrieved]

    # 5. Rank by LLM score, descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pipeline.py <path_to_resume>")
        sys.exit(1)

    ranked = run_pipeline(sys.argv[1])
    print("\n=== Ranked Job Matches ===")
    for r in ranked:
        print(f"[{r.score:3d}] {r.job_title} @ {r.company}\n      -> {r.explanation}\n")
