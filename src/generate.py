"""Prompt + thin Ollama client.

The prompt does three things:
  1. Names the role ("precise legal-document assistant")
  2. Demands a verbatim refusal sentence if the answer is missing
     from the context. The eval script and the frontend both detect
     this string.
  3. Demands bracketed citations [1] [2] so the frontend can render
     them as clickable footnotes.

Why a 7B model needs this much scaffolding: it will happily answer
from prior knowledge. The AWS agreement is public, so without an
explicit "use ONLY the context" rule the model can drift off-doc.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass

import requests

from .retrieve import RetrievedChunk, retrieve

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
TOP_K = int(os.environ.get("TOP_K", "4"))
REFUSAL_PREFIX = "The provided document does not contain information about"

PROMPT_TEMPLATE = """You are a precise legal-document assistant for the AWS Customer Agreement.
Use ONLY the numbered excerpts below. Do not use any outside knowledge.
If the answer is not contained in the excerpts, reply with exactly this sentence:
"{refusal_prefix} <topic of the question>."

Context:
{context}

Question: {question}

Answer in 2 to 4 short sentences. Cite every factual claim with the bracket number, e.g. [1].
If you are not sure, say you do not have enough information.
""".strip()


class OllamaUnavailable(RuntimeError):
    """Server is down, wrong URL, or first-call load is still in progress."""


class OllamaModelMissing(RuntimeError):
    """Server is up but the requested model tag isn't pulled."""


@dataclass
class GenerationResult:
    answer: str
    sources: list[RetrievedChunk]
    cited_chunk_ids: list[int]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = f"(page {c.page_number}"
        if c.section_heading:
            meta += f", {c.section_heading}"
        meta += ")"
        parts.append(f"[{i}] {meta}\n{c.text}")
    return "\n\n".join(parts)


def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return PROMPT_TEMPLATE.format(
        refusal_prefix=REFUSAL_PREFIX,
        context=_format_context(chunks),
        question=question.strip(),
    )


def _call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 400},
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError as e:
        raise OllamaUnavailable(
            f"Could not connect to Ollama at {OLLAMA_URL}. "
            f"Start it in another terminal with:  ollama serve"
        ) from e
    except requests.exceptions.Timeout as e:
        raise OllamaUnavailable(
            f"Ollama at {OLLAMA_URL} timed out. The first call can take 30-60s "
            f"while the model loads into RAM; later calls are fast."
        ) from e

    if response.status_code == 404:
        raise OllamaModelMissing(
            f"Ollama is running but model {OLLAMA_MODEL!r} is not pulled. "
            f"Run:  ollama pull {OLLAMA_MODEL}"
        )
    if not response.ok:
        try:
            detail = response.json().get("error") or response.text
        except Exception:
            detail = response.text
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {detail}")

    return (response.json().get("response") or "").strip()


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[int]:
    cited: list[int] = []
    seen: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        try:
            pos = int(m.group(1))
        except ValueError:
            continue
        if 1 <= pos <= len(chunks):
            cid = chunks[pos - 1].chunk_id
            if cid not in seen:
                cited.append(cid)
                seen.add(cid)
    return cited


def generate_answer(question: str, top_k: int = TOP_K) -> GenerationResult:
    chunks = retrieve(question, top_k=top_k)
    if not chunks:
        return GenerationResult(
            answer=f"{REFUSAL_PREFIX} the requested topic.",
            sources=[],
            cited_chunk_ids=[],
        )

    prompt = _build_prompt(question, chunks)
    answer = _call_ollama(prompt)
    return GenerationResult(
        answer=answer,
        sources=chunks,
        cited_chunk_ids=_extract_citations(answer, chunks),
    )


def is_refusal(answer: str) -> bool:
    return answer.strip().lower().startswith(REFUSAL_PREFIX.lower())


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What happens if the customer violates the agreement?"
    result = generate_answer(q)
    print("ANSWER:\n" + result.answer + "\n")
    print("SOURCES:")
    for i, c in enumerate(result.sources, start=1):
        print(f"  [{i}] page {c.page_number}  score={c.score:.3f}  section={c.section_heading!r}")
    print(f"\nCITED CHUNK IDS: {result.cited_chunk_ids}")
