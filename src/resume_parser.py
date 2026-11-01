"""
resume_parser.py
----------------
Extracts plain text from a resume file (.pdf, .docx, .txt) so it can be
embedded and passed to the LLM for matching.
"""

from pathlib import Path
import logging

from pypdf import PdfReader
from docx import Document

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def parse_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    text = []
    for page in reader.pages:
        content = page.extract_text() or ""
        text.append(content)
    return "\n".join(text)


def parse_docx(path: Path) -> str:
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def parse_txt(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def parse_resume(path: str) -> str:
    """Dispatch to the right parser based on file extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".pdf":
        text = parse_pdf(p)
    elif suffix == ".docx":
        text = parse_docx(p)
    elif suffix == ".txt":
        text = parse_txt(p)
    else:
        raise ValueError(f"Unsupported resume format: {suffix}. Use .pdf, .docx, or .txt")

    text = " ".join(text.split())  # collapse whitespace
    if not text.strip():
        log.warning("Extracted resume text is empty — is the PDF a scanned image? Consider OCR.")
    return text


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python resume_parser.py <path_to_resume>")
        sys.exit(1)
    print(parse_resume(sys.argv[1])[:1000])
