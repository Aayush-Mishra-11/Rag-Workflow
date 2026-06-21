"""Top-level RAG entry point. Thin on purpose so each request is obvious:

    retrieve -> build prompt -> call LLM -> extract citations -> log

The FastAPI layer and the CLI both call this.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from .db import log_query
from .generate import (
    REFUSAL_PREFIX,
    GenerationResult,
    _build_prompt,
    _call_ollama,
    _extract_citations,
    generate_answer,
    retrieve,
)


@dataclass
class RagAnswer:
    question: str
    answer: str
    sources: list[dict]
    cited_chunk_ids: list[int]
    latency_ms: int
    refused: bool
    db_id: int | None


def _refused(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_PREFIX.lower())


def _run(question: str, top_k: int) -> GenerationResult:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return GenerationResult(
            answer=f"{REFUSAL_PREFIX} the requested topic.",
            sources=[],
            cited_chunk_ids=[],
        )
    prompt = _build_prompt(question, chunks)
    text = _call_ollama(prompt)
    return GenerationResult(answer=text, sources=chunks, cited_chunk_ids=_extract_citations(text, chunks))


def answer(question: str, top_k: int | None = None) -> RagAnswer:
    t0 = time.perf_counter()
    result = _run(question, top_k=top_k or _default_top_k())
    latency_ms = int((time.perf_counter() - t0) * 1000)
    refused = _refused(result.answer)

    db_id = log_query(
        question=question,
        retrieved_chunk_ids=[c.chunk_id for c in result.sources],
        scores=[round(c.score, 4) for c in result.sources],
        cited_chunk_ids=result.cited_chunk_ids,
        answer=result.answer,
        latency_ms=latency_ms,
        refused=refused,
    )

    sources_payload = [{
        "chunk_id": c.chunk_id,
        "page": c.page_number,
        "section": c.section_heading,
        "score": round(c.score, 4),
        "preview": c.text[:240] + ("..." if len(c.text) > 240 else ""),
        "text": c.text,
    } for c in result.sources]

    return RagAnswer(
        question=question,
        answer=result.answer,
        sources=sources_payload,
        cited_chunk_ids=result.cited_chunk_ids,
        latency_ms=latency_ms,
        refused=refused,
        db_id=db_id,
    )


def _default_top_k() -> int:
    from .generate import TOP_K
    return TOP_K


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How does AWS protect customer content?"
    res = answer(q)
    print(f"\nQ: {res.question}")
    print(f"\nA: {res.answer}")
    print(f"\nLatency: {res.latency_ms} ms    Refused: {res.refused}    DB row: {res.db_id}")
    print(f"\nCited chunks: {res.cited_chunk_ids}")
    print("\nSources:")
    for i, s in enumerate(res.sources, start=1):
        print(f"  [{i}] page {s['page']} score={s['score']} section={s['section']!r}")
