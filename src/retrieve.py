"""Cosine top-k retrieval over the prebuilt numpy index.

Embeddings are L2-normalized at build time, so cosine similarity
reduces to a single matrix-vector multiply. For N ~= 80, D = 384
this is microseconds on CPU; a vector DB would be overkill.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .embed import EMBED_DIM, MODEL_NAME

VECTOR_DIR = Path("vector_store")


@dataclass
class RetrievedChunk:
    chunk_id: int
    page_number: int
    section_heading: str
    text: str
    score: float


@lru_cache(maxsize=1)
def _load_index() -> np.ndarray:
    path = VECTOR_DIR / "index.npy"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python build_index.py` first.")
    return np.load(path)


@lru_cache(maxsize=1)
def _load_meta() -> list[dict]:
    path = VECTOR_DIR / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python build_index.py` first.")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def retrieve(query: str, top_k: int = 4) -> list[RetrievedChunk]:
    if not query.strip():
        return []

    index = _load_index()
    meta = _load_meta()

    query_vec = _get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32)

    if query_vec.shape[1] != EMBED_DIM:
        raise RuntimeError(
            f"Query embedding dim {query_vec.shape[1]} != index dim {EMBED_DIM}"
        )

    scores = (query_vec @ index.T).flatten()
    top_indices = np.argsort(-scores)[:top_k]

    return [
        RetrievedChunk(
            chunk_id=m["chunk_id"],
            page_number=m["page_number"],
            section_heading=m.get("section_heading", ""),
            text=m["text"],
            score=float(scores[i]),
        )
        for i in top_indices
        for m in [meta[int(i)]]
    ]


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What happens if I breach the agreement?"
    for h in retrieve(q, top_k=4):
        print(f"[{h.chunk_id}] page={h.page_number} score={h.score:.3f} section={h.section_heading!r}")
        print(f"      {h.text[:200]}...")
        print()
