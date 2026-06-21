"""FastAPI app: serve the static frontend, expose /ask and /history."""
from __future__ import annotations

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .db import init_db, recent_queries
from .generate import OLLAMA_MODEL, OLLAMA_URL, OllamaModelMissing, OllamaUnavailable
from .rag import answer
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Vestaff RAG — AWS Customer Agreement", version="1.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=8)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        result = answer(req.question, top_k=req.top_k)
    except FileNotFoundError as e:
        # Index not built yet
        raise HTTPException(status_code=503, detail=str(e))
    except (OllamaUnavailable, OllamaModelMissing) as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {e}")

    return {
        "question": result.question,
        "answer": result.answer,
        "sources": result.sources,
        "cited_chunk_ids": result.cited_chunk_ids,
        "latency_ms": result.latency_ms,
        "refused": result.refused,
        "db_id": result.db_id,
    }


@app.get("/history")
def history(limit: int = 10) -> dict:
    return {"queries": recent_queries(limit=limit)}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ollama_url": OLLAMA_URL, "ollama_model": OLLAMA_MODEL}


@app.get("/health/ollama")
def health_ollama() -> dict:
    """Probe the Ollama server so the UI (or curl) can verify it's up."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        loaded = any(n == OLLAMA_MODEL or n.startswith(OLLAMA_MODEL + ":") for n in names)
        return {"ok": True, "models": names, "requested": OLLAMA_MODEL, "model_loaded": loaded}
    except Exception as e:
        return {"ok": False, "error": str(e), "ollama_url": OLLAMA_URL}


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.get("/")
def root() -> FileResponse:
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="frontend/index.html missing")
    return FileResponse(index)
