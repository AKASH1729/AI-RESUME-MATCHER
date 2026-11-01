"""
rag_engine.py
-------------
The "Retrieval" half of RAG.

1. Embeds every scraped job description with a Hugging Face sentence-
   transformers model.
2. Stores the embeddings in a FAISS vector index.
3. Given a resume (as text), retrieves the top-K most semantically
   similar job postings.

The retrieved jobs are later fed as *context* into the LLM (see
llm_matcher.py) so the model reasons over real postings instead of
hallucinating — that's the "Augmented Generation" half.
"""

import json
import logging
from typing import List, Dict, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, TOP_K_RETRIEVAL, INDEX_FILE, INDEX_META_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class JobRAGIndex:
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        log.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.jobs: List[Dict] = []

    # ---------- Building ----------
    def build(self, jobs: List[Dict]) -> None:
        """Embed all jobs and build a cosine-similarity FAISS index."""
        if not jobs:
            raise ValueError("No jobs provided to build the index from.")

        self.jobs = jobs
        corpus = [self._job_to_text(j) for j in jobs]

        embeddings = self.model.encode(
            corpus, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
        self.index.add(embeddings.astype(np.float32))
        log.info(f"Built FAISS index with {self.index.ntotal} job postings (dim={dim}).")

    @staticmethod
    def _job_to_text(job: Dict) -> str:
        """Flatten a job dict into a single string for embedding."""
        parts = [
            job.get("title", ""),
            job.get("company", ""),
            job.get("location", ""),
            " ".join(job.get("tags", [])),
            job.get("description", ""),
        ]
        return " | ".join(p for p in parts if p)

    # ---------- Persistence ----------
    def save(self, index_path=INDEX_FILE, meta_path=INDEX_META_FILE) -> None:
        if self.index is None:
            raise RuntimeError("Index has not been built yet.")
        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.jobs, f, ensure_ascii=False)
        log.info(f"Saved index -> {index_path}, metadata -> {meta_path}")

    def load(self, index_path=INDEX_FILE, meta_path=INDEX_META_FILE) -> None:
        self.index = faiss.read_index(str(index_path))
        with open(meta_path, encoding="utf-8") as f:
            self.jobs = json.load(f)
        log.info(f"Loaded index with {self.index.ntotal} jobs.")

    # ---------- Retrieval ----------
    def retrieve(self, query_text: str, top_k: int = TOP_K_RETRIEVAL) -> List[Tuple[Dict, float]]:
        """Return the top_k (job, similarity_score) pairs most relevant to query_text."""
        if self.index is None:
            raise RuntimeError("Index has not been built or loaded yet.")

        query_vec = self.model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(query_vec.astype(np.float32), top_k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.jobs[idx], float(score)))
        return results


if __name__ == "__main__":
    from scraper import load_jobs

    jobs = load_jobs()
    if not jobs:
        print("No jobs found — run scraper.py first.")
    else:
        rag = JobRAGIndex()
        rag.build(jobs)
        rag.save()

        sample_resume = "Experienced Python backend engineer skilled in Django, REST APIs, and PostgreSQL."
        for job, score in rag.retrieve(sample_resume, top_k=3):
            print(f"[{score:.3f}] {job['title']} @ {job['company']}")
