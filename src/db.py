"""SQLite-backed query log.

One table, one row per /ask call. JSON-encoded lists keep the
schema simple for a single-actor demo; a multi-tenant deployment
would split this into a queries table + a chunk_references table.

WAL mode lets the FastAPI writer and a CLI reader (e.g. eval) coexist.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path("rag_logs.db")
_LOCK = threading.Lock()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           REAL NOT NULL,
                question     TEXT NOT NULL,
                retrieved    TEXT NOT NULL,    -- JSON: list[int]
                scores       TEXT NOT NULL,    -- JSON: list[float]
                cited        TEXT NOT NULL,    -- JSON: list[int]
                answer       TEXT NOT NULL,
                latency_ms   INTEGER NOT NULL,
                refused      INTEGER NOT NULL  -- 0/1
            )
        """)


def log_query(*, question, retrieved_chunk_ids, scores, cited_chunk_ids,
              answer, latency_ms, refused) -> int:
    init_db()  # idempotent
    with _LOCK, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO queries
               (ts, question, retrieved, scores, cited, answer, latency_ms, refused)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), question,
             json.dumps(retrieved_chunk_ids),
             json.dumps(scores),
             json.dumps(cited_chunk_ids),
             answer,
             latency_ms,
             1 if refused else 0),
        )
        return int(cur.lastrowid)


def recent_queries(limit: int = 10) -> list[dict]:
    init_db()
    with _connect() as conn:
        rows = conn.execute("""
            SELECT id, ts, question, retrieved, scores, cited, answer,
                   latency_ms, refused
              FROM queries
             ORDER BY id DESC
             LIMIT ?
        """, (limit,)).fetchall()
    return [{
        "id": r[0], "ts": r[1], "question": r[2],
        "retrieved": json.loads(r[3]), "scores": json.loads(r[4]),
        "cited": json.loads(r[5]), "answer": r[6],
        "latency_ms": r[7], "refused": bool(r[8]),
    } for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"DB ready at {DB_PATH.resolve()}")
    for row in recent_queries(5):
        print(f"  #{row['id']}  {row['question'][:60]}  ->  {row['answer'][:60]}")
