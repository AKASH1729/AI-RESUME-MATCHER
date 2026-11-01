"""
app.py
------
Streamlit front-end. This UI is now fully decoupled from the model code:
it only talks to the FastAPI service over HTTP, so the UI never imports
transformers/FAISS/etc directly.

Run the API first:
    uvicorn api:app --reload --port 8000

Then run this UI:
    streamlit run app.py
"""

import os
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Job Matcher", page_icon="🧭", layout="centered")

st.title("🧭 AI Job Scraper & Resume Matcher")
st.caption("Upload your resume — a FastAPI service runs RAG retrieval (embeddings + FAISS) "
           "and LLM scoring (Hugging Face transformers), then returns ranked matches.")


def api_get(path: str, **kwargs):
    return requests.get(f"{API_BASE_URL}{path}", timeout=30, **kwargs)


def api_post(path: str, **kwargs):
    return requests.post(f"{API_BASE_URL}{path}", timeout=300, **kwargs)


# ---------- Sidebar: connection + job data status ----------
with st.sidebar:
    st.header("Service status")
    try:
        health = api_get("/health").json()
        if health.get("rag_index_ready"):
            st.success(f"API connected — {health.get('jobs_indexed', 0)} jobs indexed")
        else:
            st.warning("API connected, but no jobs indexed yet.")
    except requests.RequestException:
        st.error(f"Can't reach API at {API_BASE_URL}. Is `uvicorn api:app` running?")

    if st.button("🔄 Refresh job listings (scrape now)"):
        with st.spinner("Calling POST /jobs/scrape..."):
            try:
                resp = api_post("/jobs/scrape")
                if resp.ok:
                    st.success(resp.json().get("message", "Done."))
                else:
                    st.error(f"Scrape failed: {resp.json().get('detail', resp.text)}")
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")

    top_k = st.slider("How many jobs to consider (RAG top-K)", 3, 15, 5)

# ---------- Main: upload + match ----------
uploaded = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])

if uploaded is not None:
    if st.button("🔍 Find matching jobs"):
        with st.spinner("Calling POST /match — running RAG retrieval + LLM scoring..."):
            try:
                files = {"resume": (uploaded.name, uploaded.getvalue())}
                resp = api_post("/match", files=files, params={"top_k": top_k})
            except requests.RequestException as e:
                st.error(f"Request failed: {e}")
                resp = None

        if resp is not None:
            if resp.ok:
                results = resp.json()
                if not results:
                    st.info("No matches returned.")
                st.subheader("Ranked Matches")
                for r in results:
                    with st.container(border=True):
                        st.markdown(f"### {r['job_title']} — {r['company']}")
                        st.progress(r["score"] / 100)
                        st.write(f"**Match score:** {r['score']}/100")
                        st.write(r["explanation"])
            else:
                st.error(f"API error: {resp.json().get('detail', resp.text)}")
else:
    st.info("Upload a .pdf, .docx, or .txt resume to get started.")
