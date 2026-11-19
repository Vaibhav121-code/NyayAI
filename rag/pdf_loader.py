import logging
from pathlib import Path
from typing import Any

import pymupdf as fitz

logger = logging.getLogger(__name__)


def load_pdf(path: str | Path) -> dict[str, Any]:
    """Open a PDF and return a structured representation of its extracted text."""
    pdf_path = Path(path)
    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        logger.exception("Failed to open PDF: %s", pdf_path)
        return {"text": "", "pages": [], "error": str(exc), "extractable": False}

    pages: list[dict[str, Any]] = []
    all_text: list[str] = []

    try:
        for page_number in range(document.page_count):
            page = document[page_number]
            page_text = page.get_text("text")
            cleaned_text = page_text.strip()
            all_text.append(cleaned_text)
            pages.append({
                "page_number": page_number + 1,
                "text": cleaned_text,
            })
    except Exception as exc:
        logger.exception("Failed while extracting text from PDF: %s", pdf_path)
        return {
            "text": "\n\n".join(all_text),
            "pages": pages,
            "error": str(exc),
            "extractable": bool("".join(all_text).strip()),
        }
    finally:
        document.close()

    combined = "\n\n".join(page["text"] for page in pages if page["text"].strip())
    return {
        "text": combined,
        "pages": pages,
        "error": None,
        "extractable": bool(combined.strip()),
    }


def extract_text(path: str | Path) -> dict[str, Any]:
    """Wrapper for loading a PDF and extracting text with page boundaries preserved."""
    return load_pdf(path)
