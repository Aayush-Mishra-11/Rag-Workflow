# Vestaff RAG — AWS Customer Agreement

A hand-built Retrieval-Augmented Generation (RAG) system that answers
questions grounded in the AWS Customer Agreement, refuses to answer when
the document does not contain the information, cites the source page
and section, and logs every query to SQLite.

Built for the **Vestaff Junior AI Developer** technical assignment.

---

## What this does

- **Loads** the AWS Customer Agreement PDF (23 pages, ~5,500 tokens)
- **Splits** it into 54 paragraph-aware chunks (~250 tokens each, with overlap)
- **Embeds** them with `all-MiniLM-L6-v2` and saves a numpy cosine index
- **Retrieves** the top-k=4 chunks for any question
- **Generates** a grounded answer with `mistral` via Ollama, refusing if the
  context doesn't contain the answer
- **Logs** every Q&A to SQLite (`rag_logs.db`) with retrieved chunk ids,
  cosine scores, citations, latency, and refusal status
- **Serves** a chat UI on `http://127.0.0.1:8000`

No LangChain, no LangGraph, no vector database — every choice is visible in
~250 lines of Python across `src/`.

---

## Live Demo -- https://aws-rag-workflow.netlify.app/

## Sample outputs

Real answers produced by the system (see `report/technical_report.md` for
full eval results):

> **Q:** What are the limitations of liability in the agreement?
>
> **A:** The limitations of liability state that neither AWS nor the
> customer will have liability for indirect, incidental, special,
> consequential or exemplary damages, loss of profits, revenues,
> customers, opportunities, or goodwill [1]. The aggregate liability
> will not exceed the amounts paid in the 12 months before the liability
> arose, except for payment obligations and any liability that cannot be
> limited under applicable law [1]. [Sources cited from page 7,
> "Limitations of Liability" section.]

> **Q:** How can the agreement be terminated?
>
> **A:** Either party may terminate for any reason with notice and account
> closure within 30 or 60 days as stated in Section 5.2(a). For cause, a
> material breach uncured for 30 days from notice is grounds for
> termination [1]. AWS may also terminate immediately for non-compliance
> with laws or governmental requests [2]. After termination, the customer
> remains responsible for all fees through the Termination Date [3].

> **Q:** What is the recipe for chicken tikka masala?
>
> **A:** The provided document does not contain information about a recipe
> for chicken tikka masala.

---

## Prerequisites

- **Python 3.10+**
- **Ollama** with the `mistral` model pulled
- ~6 GB free RAM (the model loads into RAM)

## Setup

```bash
# 1. Python deps
pip install -r requirements.txt

# 2. Install Ollama from https://ollama.com/download, then:
ollama pull mistral

# 3. Build the index (one-time, ~30 seconds)
python build_index.py
```

## Run

You need **two terminals**:

```bash
# Terminal 1: start the Ollama server (leave running)
ollama serve

# Terminal 2: start the web app
python run.py
# -> http://127.0.0.1:8000
```

Open `http://127.0.0.1:8000`, type a question. **The first click takes
30–60 s** while Mistral loads into RAM; subsequent clicks are ~3–10 s.

## Try from the command line

```bash
python -m src.rag "What does AWS do with customer content?"
```

## Evaluate

```bash
python tests/eval.py
```

Runs all 10 questions in `tests/test_set.json`, prints a summary table, and
writes per-question results to `tests/eval_results.json`.

## Inspect the query log

The UI shows the last 5 queries. To see everything:

```bash
sqlite3 rag_logs.db "SELECT id, datetime(ts,'unixepoch'), substr(question,1,60), refused, latency_ms FROM queries ORDER BY id DESC LIMIT 10;"
```

To export everything as JSON:

```bash
sqlite3 rag_logs.db ".dump queries"
```

---

## Architecture

```
                       ┌──────────────────────────────────────┐
                       │  AWS Customer Agreement.pdf (23 pp)  │
                       └────────────────────┬─────────────────┘
                                            │ pypdf
                                            ▼
                       ┌──────────────────────────────────────┐
                       │  54 chunks (~250 tokens, 50 overlap) │
                       └────────────────────┬─────────────────┘
                                            │ sentence-transformers
                                            │ all-MiniLM-L6-v2 (384-d)
                                            ▼
                       ┌──────────────────────────────────────┐
                       │  index.npy  (54 × 384 float32)       │
                       │  meta.json  (chunk_id, page, section)│
                       └────────────────────┬─────────────────┘
                                            │ cosine top-k=4
   User ── POST /ask ──► FastAPI ── retrieve.py ──► top-k chunks
                              │
                              ▼
                       generate.py (strict prompt)
                              │
                              ▼
                       Ollama /api/generate  (mistral)
                              │
                              ▼
                       { answer, [n] citations, refused? }
                              │
                              ▼
                       SQLite (rag_logs.db)   GET /history
```

---

## Design choices (one-line each)

- **No LangChain / LangGraph** — assignment grades understanding of the pipeline
- **pypdf** for parsing — pure Python, no system deps, page-numbered
- **250-token chunks with 50-token overlap** — paragraph-aware, fits MiniLM's 512-token limit, overlap carries cross-references across boundaries
- **Numbered-heading regex** for section labels — handles "9. Limitations of Liability." correctly, not fooled by substring matches
- **all-MiniLM-L6-v2** — best size/quality tradeoff at 80 MB, runs on CPU
- **numpy cosine** — no vector DB needed for 54 chunks
- **mistral via Ollama** — local, no API key, no per-query cost
- **Strict refusal prompt** — verbatim target sentence the eval can detect
- **SQLite WAL** — single-file persistence, queryable for the report

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Could not connect to Ollama` | Start `ollama serve` in another terminal |
| `model 'mistral' is not pulled` | Run `ollama pull mistral` |
| First `/ask` times out | Mistral is loading into RAM (30–60 s). Try again. |
| `index.npy not found` | Run `python build_index.py` |
| `pip install torch` is huge | That's normal. sentence-transformers depends on torch (~800 MB on CPU). |

## Project layout

```
RAG_workflow/
├── README.md                       this file
├── requirements.txt                pinned versions
├── .gitignore                      excludes build artifacts
├── build_index.py                  PDF -> chunks -> embeddings
├── run.py                          starts uvicorn
├── src/
│   ├── parse_pdf.py                PDF -> pages
│   ├── chunk.py                    pages -> chunks (250 tok, 50 overlap)
│   ├── embed.py                    chunks -> index.npy + meta.json
│   ├── retrieve.py                 cosine top-k
│   ├── generate.py                 prompt + Ollama client + citation extraction
│   ├── rag.py                      top-level `answer(question)`
│   ├── db.py                       SQLite log
│   └── api.py                      FastAPI: /ask, /history, static frontend
├── frontend/index.html             chat UI (no build step)
├── tests/
│   ├── test_set.json               10 labeled Q&A
│   └── eval.py                     retrieval + answer metrics
├── report/
│   └── technical_report.md         2-3 page write-up
└── vector_store/                   generated; gitignored
    ├── index.npy
    └── meta.json
```

See `report/technical_report.md` for the full design write-up.
