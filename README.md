# AI Job Scraper & Resume Matcher

Scrape job postings, retrieve the jobs most relevant to a resume with sentence embeddings and FAISS, then use a Hugging Face language model to produce a match score and explanation for each result.

The application has two services connected over HTTP:

```text
Streamlit UI (src/app.py)  <---->  FastAPI service (src/api.py)
                                      |
                         scraper + resume parser + RAG + LLM matcher
```

The FastAPI service loads the embedding model, FAISS index, and language model once at startup. The Streamlit UI simply calls its API, so another web or mobile client can use the same backend.

## Features

- Scrapes listings from Arbeitnow's public job-board API (no API key required)
- Accepts PDF, DOCX, and TXT resumes
- Retrieves the most relevant jobs with `sentence-transformers` and FAISS
- Scores retrieved jobs from 0 to 100 and supplies an explanation using Hugging Face Transformers
- Offers a FastAPI API, interactive Swagger documentation, a Streamlit UI, and a command-line pipeline

## Project structure

```text
.
|-- requirements.txt
|-- README.md
|-- data/
|   `-- sample_jobs.json       # versioned offline demo data
`-- src/
    |-- config.py              # paths, model names, and constants
    |-- scraper.py             # job scraping and data persistence
    |-- resume_parser.py       # PDF, DOCX, and TXT text extraction
    |-- rag_engine.py          # embeddings and FAISS retrieval
    |-- llm_matcher.py         # Hugging Face scoring and explanations
    |-- pipeline.py            # end-to-end CLI workflow
    |-- api.py                 # FastAPI service
    `-- app.py                 # Streamlit frontend
```

Runtime-generated files such as `data/jobs.json` and the FAISS index are excluded by `.gitignore`. `data/sample_jobs.json` remains available for offline demos.

## Requirements

- Python 3.10 or later recommended
- Internet access the first time models are downloaded; also required to scrape fresh Arbeitnow listings

## Setup

From the project root, create and activate a virtual environment, then install dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the application

### 1. Prepare job data

To fetch current listings, run:

```powershell
cd src
python scraper.py
```

This saves listings to `data/jobs.json`. For an offline demo, copy the bundled sample data instead:

```powershell
Copy-Item ..\data\sample_jobs.json ..\data\jobs.json
```

### 2. Start the API

From `src/`, start FastAPI:

```powershell
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

When startup completes, open [http://localhost:8000/docs](http://localhost:8000/docs) for interactive API documentation.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Service and index status |
| `GET` | `/jobs` | Indexed job metadata |
| `POST` | `/jobs/scrape` | Scrape fresh listings and rebuild the index |
| `POST` | `/match` | Upload a resume and receive ranked matches |

Example request:

```bash
curl -X POST "http://localhost:8000/match?top_k=5" \
  -F "resume=@/path/to/your_resume.pdf"
```

### 3. Start the Streamlit UI

In a second terminal, activate the same environment and run:

```powershell
cd src
streamlit run app.py
```

The UI uses `http://localhost:8000` by default. Set `API_BASE_URL` when the API runs elsewhere:

```powershell
$env:API_BASE_URL = "http://your-server:8000"
streamlit run app.py
```

Upload a PDF, DOCX, or TXT resume and select **Find matching jobs** to see ranked results.

## Command-line alternative

To run matching without FastAPI or Streamlit, provide a resume path from the `src/` directory:

```powershell
python pipeline.py path\to\your_resume.pdf
```

If no cached jobs are present, the pipeline attempts to scrape them automatically.

## Configuration

Update [src/config.py](src/config.py) to change the embedding model, language model, retrieval count, scraper page count, or data paths. The default embedding model is `sentence-transformers/all-MiniLM-L6-v2`; the default scoring model is `google/flan-t5-base`.
