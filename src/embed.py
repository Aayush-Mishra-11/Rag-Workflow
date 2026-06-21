"""Encode chunks with sentence-transformers and persist the index.

Embedding is done once at build time, not per request, because:
- the corpus is small (~80 chunks) so the model load dominates
- the runtime path only needs numpy + the saved matrix

Outputs:
  vector_store/index.npy  (N x 384 float32, L2-normalized)
  vector_store/meta.json  (parallel list of chunk metadata)
  data/chunks.jsonl       (human-readable dump for debugging)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunk import chunk_pages
from .parse_pdf import parse_pdf

VECTOR_DIR = Path("vector_store")
DATA_DIR = Path("data")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dim


def _ensure_dirs() -> None:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def build_index(pdf_path: str | Path) -> None:
    _ensure_dirs()

    print(f"[1/4] Parsing {pdf_path} ...")
    pages = parse_pdf(pdf_path)
    print(f"      -> {len(pages)} pages")

    print("[2/4] Chunking ...")
    chunks = chunk_pages(pages)
    print(f"      -> {len(chunks)} chunks")

    with (DATA_DIR / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "page_number": c.page_number,
                "section_heading": c.section_heading,
                "text": c.text,
            }, ensure_ascii=False) + "\n")

    print(f"[3/4] Loading embedder ({MODEL_NAME}) ...")
    # Lazy import so retrieval-only paths don't pay the torch import cost.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)

    print(f"[4/4] Encoding {len(chunks)} chunks ...")
    embeddings = model.encode(
        [c.text for c in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,  # so dot product == cosine similarity
        show_progress_bar=False,
    ).astype(np.float32)

    if embeddings.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"Unexpected embedding dim {embeddings.shape[1]} (expected {EMBED_DIM})"
        )

    np.save(VECTOR_DIR / "index.npy", embeddings)
    meta = [{
        "chunk_id": c.chunk_id,
        "page_number": c.page_number,
        "section_heading": c.section_heading,
        "text": c.text,
    } for c in chunks]
    (VECTOR_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone.")
    print(f"  index: {VECTOR_DIR / 'index.npy'}  shape={embeddings.shape}")
    print(f"  meta : {VECTOR_DIR / 'meta.json'}  entries={len(meta)}")
    print(f"  dump : {DATA_DIR / 'chunks.jsonl'}")


if __name__ == "__main__":
    import sys
    pdf = sys.argv[1] if len(sys.argv) > 1 else "AWS Customer Agreement.pdf"
    build_index(pdf)
