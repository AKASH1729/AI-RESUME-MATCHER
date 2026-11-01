"""
scraper.py
----------
Collects job postings and normalizes them into a common schema:

    {
        "id": str,
        "title": str,
        "company": str,
        "location": str,
        "description": str,
        "url": str,
        "tags": [str, ...],
    }

Primary source: Arbeitnow's public job-board JSON API (no key required).
A generic BeautifulSoup-based fallback scraper is included for sites that
only expose HTML — you'll need to adapt the CSS selectors to the target
site, and check that site's robots.txt / terms of service before using it.
"""

import json
import time
import logging
from typing import List, Dict

import requests
from bs4 import BeautifulSoup

from config import ARBEITNOW_API, SCRAPE_MAX_PAGES, REQUEST_TIMEOUT, REQUEST_DELAY, JOBS_FILE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _normalize_arbeitnow(raw: dict) -> Dict:
    return {
        "id": raw.get("slug", raw.get("url", "")),
        "title": raw.get("title", "").strip(),
        "company": raw.get("company_name", "").strip(),
        "location": raw.get("location", "Remote").strip() or "Remote",
        "description": BeautifulSoup(raw.get("description", ""), "html.parser").get_text(" ", strip=True),
        "url": raw.get("url", ""),
        "tags": raw.get("tags", []),
    }


def fetch_from_api(max_pages: int = SCRAPE_MAX_PAGES) -> List[Dict]:
    """Pull job listings from the Arbeitnow public API (paginated)."""
    jobs: List[Dict] = []
    page = 1
    while page <= max_pages:
        try:
            resp = requests.get(ARBEITNOW_API, params={"page": page}, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.warning(f"Request failed on page {page}: {e}")
            break

        payload = resp.json()
        batch = payload.get("data", [])
        if not batch:
            break

        jobs.extend(_normalize_arbeitnow(j) for j in batch)
        log.info(f"Fetched page {page}: {len(batch)} jobs (total so far: {len(jobs)})")

        page += 1
        time.sleep(REQUEST_DELAY)

    return jobs


def fetch_from_html(list_url: str, item_selector: str, field_selectors: Dict[str, str]) -> List[Dict]:
    """
    Generic fallback scraper for a plain HTML job listing page.

    Example:
        fetch_from_html(
            "https://example-job-board.com/jobs",
            item_selector="div.job-card",
            field_selectors={
                "title": "h2.job-title",
                "company": "span.company-name",
                "location": "span.job-location",
                "description": "div.job-summary",
                "url": "a.job-link",   # href will be pulled automatically
            },
        )

    NOTE: Always confirm the target site's robots.txt / ToS permit scraping
    before pointing this at a real site, and expect to update selectors
    whenever the site's markup changes.
    """
    jobs = []
    try:
        resp = requests.get(list_url, timeout=REQUEST_TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        log.error(f"Could not fetch {list_url}: {e}")
        return jobs

    soup = BeautifulSoup(resp.text, "html.parser")
    for i, item in enumerate(soup.select(item_selector)):
        job = {"id": f"html-{i}", "tags": []}
        for field, sel in field_selectors.items():
            node = item.select_one(sel)
            if node is None:
                job[field] = ""
            elif field == "url" and node.has_attr("href"):
                job[field] = node["href"]
            else:
                job[field] = node.get_text(" ", strip=True)
        jobs.append(job)

    return jobs


def save_jobs(jobs: List[Dict], path=JOBS_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    log.info(f"Saved {len(jobs)} jobs to {path}")


def load_jobs(path=JOBS_FILE) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    jobs = fetch_from_api()
    if jobs:
        save_jobs(jobs)
    else:
        log.warning("No jobs fetched — check network access or the API endpoint.")
