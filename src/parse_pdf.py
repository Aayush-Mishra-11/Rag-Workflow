"""PDF -> list of pages with their text and 1-based page number."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Page:
    page_number: int
    text: str


def parse_pdf(pdf_path: str | Path) -> list[Page]:
    reader = PdfReader(str(pdf_path))
    pages: list[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Page(page_number=i, text=text))
    return pages


if __name__ == "__main__":
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "AWS Customer Agreement.pdf"
    pages = parse_pdf(pdf)
    print(f"Parsed {len(pages)} pages from {pdf}")
    if pages:
        print(f"\n--- page 1 ({len(pages[0].text)} chars) ---\n")
        print(pages[0].text[:500])
