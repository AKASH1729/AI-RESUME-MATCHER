"""
llm_matcher.py
--------------
The "Generation" half of RAG.

Takes:
  - the resume text
  - one retrieved job posting (context, pulled by rag_engine.py)

...and uses a Hugging Face sequence-to-sequence model
to produce a match score (0-100) and a short natural-language
explanation of why the resume is or isn't a good fit.

Swap LLM_MODEL in config.py for a larger instruct model if you have a
GPU; flan-t5-base keeps this runnable on CPU for demo purposes.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import LLM_MODEL, LLM_MAX_NEW_TOKENS, MATCH_SCORE_MIN, MATCH_SCORE_MAX

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


@dataclass
class MatchResult:
    job_title: str
    company: str
    score: int
    explanation: str


PROMPT_TEMPLATE = """You are a recruiting assistant. Compare the candidate's resume to the job posting.

Resume:
{resume}

Job Posting:
Title: {title}
Company: {company}
Description: {description}

Task: Give a match score from 0 to 100 (how well the resume fits this job) and a one-sentence
explanation. Respond in exactly this format:
Score: <number>
Reason: <one sentence>
"""


class LLMMatcher:
    def __init__(self, model_name: str = LLM_MODEL):
        log.info(f"Loading LLM: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()

    def _truncate(self, text: str, max_chars: int = 1500) -> str:
        return text if len(text) <= max_chars else text[:max_chars] + "..."

    def match(self, resume_text: str, job: Dict) -> MatchResult:
        prompt = PROMPT_TEMPLATE.format(
            resume=self._truncate(resume_text),
            title=job.get("title", ""),
            company=job.get("company", ""),
            description=self._truncate(job.get("description", "")),
        )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=LLM_MAX_NEW_TOKENS,
                do_sample=False,
            )
        output = self.tokenizer.decode(generated[0], skip_special_tokens=True)

        score, reason = self._parse_output(output)

        return MatchResult(
            job_title=job.get("title", ""),
            company=job.get("company", ""),
            score=score,
            explanation=reason,
        )

    @staticmethod
    def _parse_output(text: str) -> (int, str):
        """Pull a numeric score and reason out of the model's raw output.
        Falls back gracefully if the model doesn't follow the format exactly."""
        score_match = re.search(r"Score:\s*(\d{1,3})", text)
        reason_match = re.search(r"Reason:\s*(.+)", text, re.DOTALL)

        score = int(score_match.group(1)) if score_match else 50
        score = max(MATCH_SCORE_MIN, min(MATCH_SCORE_MAX, score))

        reason = reason_match.group(1).strip() if reason_match else text.strip()
        return score, reason


if __name__ == "__main__":
    matcher = LLMMatcher()
    sample_job = {
        "title": "Backend Engineer",
        "company": "Acme Corp",
        "description": "We need a Python developer experienced with Django, REST APIs, and PostgreSQL.",
    }
    sample_resume = "5 years of Python backend development using Django and PostgreSQL."
    result = matcher.match(sample_resume, sample_job)
    print(result)
