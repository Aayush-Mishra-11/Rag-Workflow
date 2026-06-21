"""Paragraph-aware chunker with overlap and section labels.

Headings in the AWS Customer Agreement are formatted as numbered titles
at the start of a line, e.g. "9. Limitations of Liability.". We match
that pattern and use the heading as the chunk's section label — this is
the real label, not a guess from a fixed keyword list, so a single
chunker works on this and any other numbered-document.

Token counting uses whitespace. It's fast and dependency-free; the
MiniLM embedder re-tokenizes downstream anyway.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .parse_pdf import Page

MAX_TOKENS = 250
OVERLAP_TOKENS = 50

# Matches a numbered section heading at the start of a line, e.g.
#   "9. Limitations of Liability."
#   "5.2 Effect of Suspension."   (sub-section, no dot in number)
# We only treat it as a section when the number has no dots (top-level).
_HEADING_RE = re.compile(r"^(\d+)\.\s+([^\n]+?)\.?\s*$", re.MULTILINE)


@dataclass
class Chunk:
    chunk_id: int
    page_number: int
    section_heading: str
    text: str


def _find_section_heading(text: str) -> str:
    """Return the first top-level numbered heading in this page, or ''."""
    for m in _HEADING_RE.finditer(text):
        heading = m.group(2).strip()
        # The AWS PDF renders some words as ligatures (e.g. "Deﬁnitions"
        # instead of "Definitions"). Normalize to ASCII so the label
        # matches downstream strings and SQL rows.
        return (heading
                .replace("ﬁ", "fi")
                .replace("ﬂ", "fl")
                .replace("ﬀ", "ff")
                .replace("ﬃ", "ffi")
                .replace("ﬄ", "ffl"))
    return ""


def _split_paragraphs(page_text: str) -> list[str]:
    # pypdf gives us soft line-wrapped text (single \n), not paragraph
    # breaks. Treat every non-empty line as a paragraph so we get
    # something the chunker can actually pack.
    return [p.strip() for p in page_text.split("\n") if p.strip()]


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_id = 0
    current_section = ""

    for page in pages:
        # Only overwrite the current section if THIS page contains a
        # top-level numbered heading. Sub-section headings like
        # "5.2 Effect of Suspension." don't change the label.
        page_section = _find_section_heading(page.text)
        if page_section:
            current_section = page_section

        buffer: list[str] = []
        for para in _split_paragraphs(page.text):
            para_tokens = para.split()
            if buffer and len(buffer) + len(para_tokens) > MAX_TOKENS:
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    page_number=page.page_number,
                    section_heading=current_section,
                    text=" ".join(buffer),
                ))
                chunk_id += 1
                buffer = buffer[-OVERLAP_TOKENS:]
            buffer.extend(para_tokens)

        if buffer:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                page_number=page.page_number,
                section_heading=current_section,
                text=" ".join(buffer),
            ))
            chunk_id += 1

    return chunks


if __name__ == "__main__":
    from .parse_pdf import parse_pdf
    import sys

    pdf = sys.argv[1] if len(sys.argv) > 1 else "AWS Customer Agreement.pdf"
    pages = parse_pdf(pdf)
    chunks = chunk_pages(pages)
    print(f"Built {len(chunks)} chunks from {len(pages)} pages")
    if chunks:
        c = chunks[0]
        print(f"\n--- chunk 0 (page {c.page_number}, section={c.section_heading!r}) ---")
        print(c.text[:400])