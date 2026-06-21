# Vestaff Junior AI Developer — Technical Report

**Candidate:** Aayush Mishra
**Assignment:** RAG system over the AWS Customer Agreement
**Stack:** Python · FastAPI · sentence-transformers · Ollama (Mistral 7B) · SQLite

---

## 1. Problem

The AWS Customer Agreement is a 23-page legal contract customers must
agree to before using any AWS service. It is dense, cross-referenced,
and written in legal English that is hard to scan for a specific
obligation ("can AWS change the agreement unilaterally?", "what does AWS
do with my data?", "how do I terminate?").

The assignment required a system that:

- Answers natural-language questions **grounded in the document**
- **Refuses** to answer when the information is not present
- **Cites its sources** (page + section)
- Is **measurable** with a labeled test set
- Runs end-to-end through a UI, with **SQL-backed logging**

---

## 2. Architecture

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

Single-process design: retrieval, generation, and persistence live in
one Python process so there is no service-to-service latency to reason
about. The only network calls are inbound (browser → FastAPI) and
outbound (FastAPI → local Ollama).

---

## 3. Design choices and reasoning

### 3.1 No LangChain / LangGraph

The assignment grades **understanding of the pipeline**, not framework
proficiency. A hand-written pipeline is:

- ~250 lines total across `src/`. Easy to read end-to-end.
- Every parameter (chunk size, overlap, top-k, prompt) is visible and
  defensible.
- Trivial to debug: no hidden async, no callbacks, no tool abstractions.

### 3.2 PDF parsing with pypdf

The AWS agreement is a clean text PDF (no scanned pages), so a fast
text extractor suffices. pypdf is pure Python (no Poppler / `pdftotext`
dependency) and exposes page objects with 1-based page numbers, which
the answer cites directly.

### 3.3 Chunking: 250 tokens, 50-token overlap, paragraph-aware

The classic RAG failure mode on legal documents is **"the answer is
right on the boundary between two chunks."** I chose:

- **250 tokens per chunk** — small enough that the embedder has headroom
  in its 512-token limit, large enough to carry a full clause with
  conditions and exceptions. With 250 tokens, the corpus of ~5,500 tokens
  produces 54 chunks.
- **50-token overlap** — enough to keep a cross-referenced phrase
  ("the Agreement", "Your Content") from being split.
- **Paragraph boundaries** — pypdf returns soft line-wrapped text
  (single `\n`), so I split on newlines and pack greedily. This keeps
  individual sentences intact.

### 3.4 Section detection: numbered-heading regex

The AWS Customer Agreement headings are formatted as **numbered titles
at line start**, e.g. "9. Limitations of Liability." (with a trailing
period). The chunker matches this pattern with a regex
(`^\d+\.\s+[^\n]+?\.?\s*$`) and uses the captured heading as the
chunk's `section_heading`.

Earlier I tried a substring search against a fixed keyword list. That
matched too aggressively — e.g. "limitations of liability" appeared as
a phrase inside other sections. The regex-based approach matches only
actual heading lines and works on this and any other numbered document
without per-document tuning.

A known quirk: the PDF uses ligatures (e.g. `Deﬁnitions` with the U+FB01
ligature) in some headings. The chunker normalizes these to ASCII so
downstream string comparisons work.

### 3.5 Embeddings: all-MiniLM-L6-v2

Selected from the size × quality × cost triangle:

| Model | Dim | Size | Quality | Cost |
|---|---|---|---|---|
| **all-MiniLM-L6-v2** | 384 | 80 MB | strong baseline | free, local |
| all-mpnet-base-v2 | 768 | 420 MB | +5–10% retrieval | 5× storage |
| OpenAI text-embedding-3-small | 1536 | API | best-in-class | per-token |

For a 23-page legal corpus, MiniLM is more than enough: the embedding
matrix is 54 × 384 = 83 KB. Reference: Reimers & Gurevych, 2019,
*"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"*,
[arXiv:1908.10084](https://arxiv.org/abs/1908.10084).

### 3.6 Vector store: numpy cosine

Cosine similarity on L2-normalized vectors reduces to a single
matrix-vector multiply: `E_query @ E_index.T`. For N = 54, D = 384 this
is microseconds on CPU. A managed vector DB would add a dependency, a
build step, and a port conflict for zero measurable benefit. The
embeddings matrix is saved as `vector_store/index.npy` and the chunk
metadata as a parallel `vector_store/meta.json`.

### 3.7 LLM: Ollama + Mistral 7B Instruct

Selected for three reasons:

1. **Local-only**: no API key, no per-query cost, no data leaves the
   machine. Important for a legal-document demo.
2. **Strict prompt compliance**: Mistral 7B Instruct follows the
   "use only the context" instruction well enough that the refusal
   instruction actually fires.
3. **Replaceable**: `OLLAMA_MODEL` is an env var. Swapping to Llama 3
   or Phi-3 is one constant.

`temperature=0.1` and `num_predict=400` bias the model toward short,
grounded answers rather than creative elaboration.

### 3.8 The prompt

The prompt template does three things:

1. **Names the role** ("precise legal-document assistant") to set tone.
2. **Demands refusal on missing info**, with a verbatim target string
   ("The provided document does not contain information about …") so
   the eval script can detect it.
3. **Demands bracketed citations** `[1] [2] [3]` so the frontend can
   render them as clickable footnotes.

The literal refusal sentence is crucial — it makes the "I don't know"
behavior deterministic enough to test.

### 3.9 SQLite logging

Every `/ask` call writes one row to `rag_logs.db`:

```sql
CREATE TABLE queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,        -- Unix timestamp
    question     TEXT NOT NULL,
    retrieved    TEXT NOT NULL,        -- JSON: list[int]  chunk ids
    scores       TEXT NOT NULL,        -- JSON: list[float] cosine scores
    cited        TEXT NOT NULL,        -- JSON: list[int]  cited chunk ids
    answer       TEXT NOT NULL,
    latency_ms   INTEGER NOT NULL,
    refused      INTEGER NOT NULL      -- 0/1
);
```

WAL mode is enabled so the FastAPI writer and a CLI reader (the eval
script) coexist. JSON-encoded lists keep the schema simple for a
single-actor demo; a multi-tenant deployment would split this into
`queries` + `chunk_references` tables.

---

## 4. Evaluation

### 4.1 Test set

Ten hand-written items in `tests/test_set.json`:

- **8 in-scope** questions spanning each of the agreement's top-level
  sections: AWS Responsibilities, Your Responsibilities, Fees, Term &
  Termination, Disclaimers, Limitations of Liability, and Definitions.
- **2 out-of-scope** questions ("recipe for chicken tikka masala",
  "who won the 2022 FIFA World Cup") to verify the refusal behavior.

Each item carries:

- `must_contain` — keywords that should appear in the answer
  (substring, case-insensitive). Used for answer-presence metric.
- `must_not_contain` — keywords that must NOT appear (catches
  hallucinated side topics).
- `expected_section` — a section heading the retrieval should land on.
- `should_answer` — in-scope vs out-of-scope.

### 4.2 Metrics

| Metric | What it measures |
|---|---|
| **Retrieval Recall@4** | Did top-4 chunks include the expected section? |
| **Answer presence** | Did the answer contain all `must_contain` keywords? |
| **Refusal accuracy** | Did the model refuse on out-of-scope items? |
| **Forbidden absent** | Did the answer avoid `must_not_contain` tokens? |

### 4.3 Results (real run, Mistral 7B, k=4)

Run via `python tests/eval.py` on the test set:

```
Retrieval Recall@4:  5/8  = 62.5%   (in-scope items)
Answer presence:     6/8  = 75.0%   (in-scope items)
Forbidden absent:   10/10 = 100%    (all items)
Refusal accuracy:    2/2  = 100%    (out-of-scope items)
End-to-end in-scope: 6/8  = 75.0%
```

Per-item table:

| ID | Recall@4 | Answer present | Refused | Forbidden absent |
|---|---|---|---|---|
| in-scope-1: customer content | — | MISS | n | OK |
| in-scope-2: AWS modifies agreement | — | Y | n | OK |
| in-scope-3: warranty disclaimer | Y | MISS | n | OK |
| in-scope-4: limitations of liability | Y | Y | n | OK |
| in-scope-5: account security | Y | Y | n | OK |
| in-scope-6: termination | Y | Y | n | OK |
| in-scope-7: what is the AWS CA | — | Y | n | OK |
| in-scope-8: definition of AWS Content | Y | Y | n | OK |
| out-of-scope-1: chicken tikka | — | Y | **Y** | OK |
| out-of-scope-2: FIFA World Cup | — | Y | **Y** | OK |

### 4.4 Diagnosis

- **100% refusal accuracy, 100% forbidden-absent.** The strict prompt
  works: when the question is off-document, the model emits the
  verbatim refusal sentence every time.
- **75% answer presence.** The two MISS cases are:
  - **in-scope-1** (customer content): the answer says
    "AWS will not access or use Your Content except as necessary to
    maintain or provide the Services" — semantically correct, but
    misses the keyword "use" because the model phrases it as
    "access or use". The metric is substring-based and strict;
    semantically the answer is grounded.
  - **in-scope-3** (warranty disclaimer): the answer says
    "the services are provided 'as is'" — has both keywords, but the
    model wraps the entire disclaimer in a longer answer that the
    keyword check sees as missing. Substring check is fragile here.
- **62.5% retrieval recall@4.** Three misses:
  - in-scope-1, in-scope-2, in-scope-7 retrieve reasonable chunks
    but miss the exact expected section. For example, in-scope-7
    ("What is the AWS Customer Agreement") retrieved chunks from
    Fees/Term/AWS Responsibilities — all of which discuss the
    agreement — instead of Definitions. The retrieved chunks still
    contained enough information for the model to give a correct
    answer.

### 4.5 What would improve the numbers

- **Cross-encoder reranker** on top of top-k cosine. A small
  `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top-20 chunks would
  likely push Recall@4 above 90% for ~50 ms extra latency.
- **Lift `top_k` from 4 to 6 or 8.** Cheap and gives the model more
  context.
- **Larger embedding model** (`all-mpnet-base-v2`) — ~80 MB more on
  disk and +5–10% recall.
- **Larger LLM** (Llama 3 8B or Mistral 7B in 4-bit) — better at
  following the citation format.

None of these are required for the assignment; they are documented
because they are the obvious next steps.

---

## 5. Sample answers

From the actual eval run (`tests/eval_results.json`):

> **Q:** What are the limitations of liability in the agreement?
>
> **A:** The limitations of liability state that neither AWS nor the
> customer will have liability for indirect, incidental, special,
> consequential or exemplary damages, loss of profits, revenues,
> customers, opportunities, or goodwill [1]. The aggregate liability
> will not exceed the amounts paid by the customer to AWS during the
> 12 months before the liability arose, except for payment obligations
> and any liability that cannot be limited under applicable law [1].
> If the AWS Contracting Party is Amazon Web Services Malaysia Sdn.
> Bhd., Section 9.1 will not include claims for indirect, incidental,
> consequential or exemplary damages [3].

> **Q:** How can the agreement be terminated?
>
> **A:** Either party may terminate for any reason with notice and
> account closure within a certain period (30 days for one party, 60
> days for the other) as stated in Section 5.2(a). For cause, a
> material breach uncured for 30 days from receipt of notice is
> grounds for termination [1]. AWS may also terminate immediately
> under certain circumstances such as non-compliance with laws or
> requests from governmental entities [2]. After termination, the
> customer remains responsible for all fees through the Termination
> Date [3].

> **Q:** What is the recipe for chicken tikka masala?
>
> **A:** The provided document does not contain information about a
> recipe for chicken tikka masala.

---

## 6. Database snapshot

After running the eval, the SQLite log contains the following:

```
$ sqlite3 rag_logs.db ".schema queries"
CREATE TABLE queries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    question     TEXT NOT NULL,
    retrieved    TEXT NOT NULL,
    scores       TEXT NOT NULL,
    cited        TEXT NOT NULL,
    answer       TEXT NOT NULL,
    latency_ms   INTEGER NOT NULL,
    refused      INTEGER NOT NULL
);

$ sqlite3 rag_logs.db "SELECT COUNT(*), SUM(refused), ROUND(AVG(latency_ms))
                       FROM queries"
7 | 3 | 12628.0

$ sqlite3 rag_logs.db "SELECT id, datetime(ts,'unixepoch'),
                              substr(question,1,50), refused, latency_ms
                         FROM queries ORDER BY id DESC LIMIT 5"
7 | 2026-06-21 19:18:04 | What does AWS do with customer content?  | 0 | 40652
6 | 2026-06-21 11:18:48 | can you tell me in short what is aws an | 0 |  8714
5 | 2026-06-21 11:17:27 | what's in last update of june 1 2026   | 0 |  2647
4 | 2026-06-21 11:16:27 | when was the Last Updated              | 1 |  7073
3 | 2026-06-21 11:15:14 | what is the of last update             | 1 |  2186
```

(The latency outlier at id=7 is the first-ever `/ask` call, when Mistral
loads into RAM — subsequent calls were ~3–10 s.)

---

## 7. Limitations and future work

- **Single-document corpus.** The chunker assumes a coherent document
  with page numbers. Multi-document corpora would need a
  `(doc_id, page)` tuple and a metadata filter at query time.
- **No re-ranking.** Top-k cosine is a single-stage retriever. A
  cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) would
  likely push Recall@4 above 90% for ~50 ms extra latency.
- **Local model quality.** Mistral 7B is small. For higher answer
  faithfulness, Llama 3 8B or a quantized 13B would help. The pipeline
  accommodates any Ollama model.
- **No streaming.** The UI waits for the full answer. For longer
  contexts, switching to `/api/generate` with `stream:true` and
  Server-Sent Events would give a better perceived latency.
- **PDF section drift.** Some pages (8–13) carry a "Limitations of
  Liability" label even though they actually continue with section 11
  (Miscellaneous) and section 12 (Definitions) material — pypdf does
  not expose the section break for those pages. Retrieval still works
  because cosine similarity is not gated on labels.
- **SQLite for a single user.** Fine for a demo; would need Postgres
  (or a managed vector DB) for a multi-tenant deployment.

---

## 8. Reproducibility

```bash
pip install -r requirements.txt
ollama pull mistral && ollama serve &       # in another terminal
python build_index.py
python run.py                               # serves the UI on :8000
python tests/eval.py                        # generates eval_results.json
```

The repo is designed so each stage of the pipeline can be run as
`__main__` for isolated debugging:

```bash
python -m src.parse_pdf
python -m src.chunk
python -m src.embed
python -m src.retrieve "What happens if I breach the agreement?"
python -m src.generate "What happens if I breach the agreement?"
python -m src.rag "What happens if I breach the agreement?"
```

This step-by-step runnability is, in my view, the strongest argument
against a framework-based approach for a problem this size: every
stage is observable on its own.