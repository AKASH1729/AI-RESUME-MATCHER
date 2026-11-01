"""
api.py
------
FastAPI service that wraps the scraping / RAG / LLM pipeline behind a
clean HTTP API. This decouples the UI (Streamlit, a web frontend, a
mobile app, etc.) from the model code — the UI just calls endpoints.

Heavy objects (embedding model, LLM, FAISS index) are loaded ONCE at
startup and kept in memory, so requests stay fast after the first load.

Run with:
    uvicorn api:app --reload --host 0.0.0.0 --port 8000

Interactive API docs are then available at http://localhost:8000/docs
"""

import logging
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TOP_K_RETRIEVAL
from scraper import fetch_from_api, save_jobs, load_jobs
from resume_parser import parse_resume
from rag_engine import JobRAGIndex
from llm_matcher import LLMMatcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------- In-memory model registry ----------
class ModelState:
    rag: Optional[JobRAGIndex] = None
    matcher: Optional[LLMMatcher] = None


state = ModelState()


def build_index_from_available_jobs() -> None:
    jobs = load_jobs()
    if not jobs:
        log.info("No cached jobs on disk — scraping now...")
        jobs = fetch_from_api()
        if jobs:
            save_jobs(jobs)

    if not jobs:
        log.warning("No jobs available yet. Call POST /jobs/scrape once you have network access.")
        state.rag = None
        return

    rag = JobRAGIndex()
    rag.build(jobs)
    state.rag = rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models once
    log.info("Loading LLM matcher...")
    state.matcher = LLMMatcher()
    log.info("Building RAG index from available jobs...")
    build_index_from_available_jobs()
    log.info("Startup complete.")
    yield
    # Shutdown: nothing to clean up explicitly
    log.info("Shutting down API.")


app = FastAPI(
    title="AI Job Scraper & Resume Matcher API",
    description="RAG + Hugging Face transformers service connecting job data, "
                 "resumes, and match scoring to any front-end UI.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow any frontend (Streamlit, a web app, etc.) to call this API during development.
# Lock this down to specific origins before deploying publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Schemas ----------
class JobOut(BaseModel):
    id: str
    title: str
    company: str
    location: str
    url: str
    tags: List[str]


class MatchOut(BaseModel):
    job_title: str
    company: str
    score: int
    explanation: str


class ScrapeResponse(BaseModel):
    jobs_found: int
    message: str


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_index_ready": state.rag is not None,
        "jobs_indexed": state.rag.index.ntotal if state.rag and state.rag.index else 0,
    }


@app.get("/jobs", response_model=List[JobOut])
def list_jobs(limit: int = Query(50, ge=1, le=500)):
    """Return currently indexed jobs (metadata only)."""
    if not state.rag or not state.rag.jobs:
        raise HTTPException(status_code=404, detail="No jobs indexed yet. Call POST /jobs/scrape first.")
    return state.rag.jobs[:limit]


@app.post("/jobs/scrape", response_model=ScrapeResponse)
def scrape_jobs():
    """Trigger a fresh scrape and rebuild the RAG index with the new data."""
    jobs = fetch_from_api()
    if not jobs:
        raise HTTPException(status_code=502, detail="Scrape failed — check network access to the job API.")

    save_jobs(jobs)
    rag = JobRAGIndex()
    rag.build(jobs)
    state.rag = rag

    return ScrapeResponse(jobs_found=len(jobs), message="Jobs scraped and index rebuilt.")


@app.post("/match", response_model=List[MatchOut])
async def match_resume(
    resume: UploadFile = File(..., description="Resume file: .pdf, .docx, or .txt"),
    top_k: int = Query(TOP_K_RETRIEVAL, ge=1, le=20, description="How many jobs to retrieve and score"),
):
    """
    Upload a resume and get back a ranked list of job matches.

    This is the main endpoint the UI calls: it runs RAG retrieval
    (embeddings + FAISS) followed by LLM scoring for each retrieved job.
    """
    if state.rag is None:
        raise HTTPException(
            status_code=503,
            detail="No job index available yet. Call POST /jobs/scrape first.",
        )

    suffix = Path(resume.filename).suffix.lower()
    if suffix not in (".pdf", ".docx", ".txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await resume.read())
        tmp_path = tmp.name

    try:
        resume_text = parse_resume(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse resume: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    if not resume_text.strip():
        raise HTTPException(status_code=422, detail="Resume text extraction returned empty content.")

    retrieved = state.rag.retrieve(resume_text, top_k=top_k)
    results = [state.matcher.match(resume_text, job) for job, _sim in retrieved]
    results.sort(key=lambda r: r.score, reverse=True)

    return [
        MatchOut(job_title=r.job_title, company=r.company, score=r.score, explanation=r.explanation)
        for r in results
    ]
