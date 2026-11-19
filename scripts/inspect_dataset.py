from __future__ import annotations

import logging
import sys
from pathlib import Path

import pymupdf as fitz

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_pdfs(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Data directory does not exist: {directory}")
    return sorted(directory.rglob("*.pdf"))


def inspect_pdf(pdf_path: Path) -> dict[str, object]:
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        extracted_chars = 0
        extractable = False
        for page_index in range(page_count):
            page = doc[page_index]
            text = page.get_text("text")
            if text:
                extractable = True
                extracted_chars += len(text)
        doc.close()
        return {
            "pages": page_count,
            "characters": extracted_chars,
            "extractable": extractable,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "pages": 0,
            "characters": 0,
            "extractable": False,
            "error": str(exc),
        }


def main() -> None:
    print("Dataset inspection")
    print("------------------")

    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        print("Add PDFs to data/raw/ before running ingestion or search.")
        return

    pdf_files = find_pdfs(DATA_DIR)
    print(f"PDF files found: {len(pdf_files)}")

    if not pdf_files:
        print(f"No PDFs were found in {DATA_DIR}.")
        print("Place local judgments in data/raw/ and run the script again.")
        return

    readable_count = 0
    unreadable_count = 0

    for pdf_file in pdf_files:
        print(f"\n{pdf_file.name}")
        result = inspect_pdf(pdf_file)
        pages = result["pages"]
        characters = result["characters"]
        extractable = result["extractable"]
        error = result["error"]

        if error is None and extractable:
            readable_count += 1
            print(f"Pages: {pages}")
            print(f"Characters: {characters}")
            print("Extractable: YES")
        else:
            unreadable_count += 1
            print(f"Pages: {pages}")
            print(f"Characters: {characters}")
            print("Extractable: NO")
            if error:
                print(f"Issue: {error}")

    print("\nSummary:")
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Readable PDFs: {readable_count}")
    print(f"Unreadable PDFs: {unreadable_count}")


if __name__ == "__main__":
    main()
