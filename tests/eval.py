"""Hand-built evaluation harness.

What this measures (and what it deliberately does not):

1. **Retrieval Recall@4** — for each in-scope question, do the top-4
   retrieved chunks include a chunk whose `section_heading` matches
   the expected section? This is a coarse but reasonable proxy for
   "the answer's source was available in the top-k context block."

2. **Answer presence** — for in-scope items, do all `must_contain`
   tokens (case-insensitive, substring) appear in the generated
   answer? A loose check; the small model may paraphrase, but the
   core vocabulary should land.

3. **Refusal accuracy** — for out-of-scope items, does the model
   emit our literal refusal prefix?

4. **Forbidden content** — `must_not_contain` tokens must NOT appear
   in the answer. Useful for catching hallucinated side topics.

Reports are printed and also written to `tests/eval_results.json` so
the technical report can quote them.

Note: this script calls Ollama, so it requires `ollama serve` to be
running with `mistral` available. Pass `--no-llm` to skip generation
and only measure retrieval recall (faster, no model needed).
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_SET = Path(__file__).resolve().parent / "test_set.json"
RESULTS = Path(__file__).resolve().parent / "eval_results.json"


def _load_test_set() -> list[dict]:
    data = json.loads(TEST_SET.read_text(encoding="utf-8"))
    # Accept either a bare list or {"items": [...]} for forward compatibility.
    if isinstance(data, dict) and "items" in data:
        return data["items"]
    return data


def _norm(s: str) -> str:
    """ASCII-fold so PDF ligatures ('Deﬁnitions') match plain ASCII."""
    return (s.replace("ﬁ", "fi").replace("ﬂ", "fl")
             .replace("ﬀ", "ff").replace("ﬃ", "ffi").replace("ﬄ", "ffl")
             .lower())


def _has_section(chunks, section: str) -> bool:
    if not section:
        return False
    target = _norm(section)
    return any(_norm(c.get("section_heading") or "") == target for c in chunks)


def _evaluate_retrieval(item: dict, top_chunks: list[dict]) -> dict:
    expected = item.get("expected_section", "")
    hit = _has_section(top_chunks, expected)
    return {
        "expected_section": expected,
        "retrieved_sections": [c.get("section_heading") for c in top_chunks],
        "recall_at_4": hit,
    }


def _has_all(text: str, tokens: list[str]) -> bool:
    lower = text.lower()
    return all(tok.lower() in lower for tok in tokens)


def _has_any(text: str, tokens: list[str]) -> bool:
    lower = text.lower()
    return any(tok.lower() in lower for tok in tokens)


def _is_refusal(text: str) -> bool:
    return text.strip().lower().startswith(
        "the provided document does not contain information about"
    )


def _evaluate_answer(item: dict, answer: str) -> dict:
    out: dict = {"refused": _is_refusal(answer)}
    must_contain = item.get("must_contain", [])
    must_not_contain = item.get("must_not_contain", [])
    out["must_contain_met"] = _has_all(answer, must_contain) if must_contain else True
    out["must_not_contain_met"] = (
        not _has_any(answer, must_not_contain) if must_not_contain else True
    )
    out["present_tokens"] = [t for t in must_contain if t.lower() in answer.lower()]
    out["missing_tokens"] = [t for t in must_contain if t.lower() not in answer.lower()]
    return out


def main(include_llm: bool, top_k: int = 4) -> None:
    items = _load_test_set()
    results: list[dict] = []

    # Lazy imports so --no-llm mode doesn't need sentence-transformers
    # at the top of the file which would slow --help.
    from src.retrieve import retrieve
    if include_llm:
        from src.generate import generate_answer

    print(f"Evaluating {len(items)} items  (llm={include_llm}, top_k={top_k})\n")
    print(f"{'id':22} {'recall@4':10} {'ans':12} {'refus':6} {'forb':6}  q")
    print("-" * 100)

    in_scope = [it for it in items if it.get("should_answer")]
    out_of_scope = [it for it in items if not it.get("should_answer")]

    for item in items:
        q = item["question"]
        chunks = retrieve(q, top_k=top_k)
        # serialize chunks for eval_results
        chunk_meta = [
            {
                "chunk_id": c.chunk_id,
                "page": c.page_number,
                "section_heading": c.section_heading,
                "score": round(c.score, 4),
            }
            for c in chunks
        ]
        retrieval_eval = _evaluate_retrieval(item, chunk_meta)

        if include_llm:
            t0 = time.perf_counter()
            try:
                gen = generate_answer(q, top_k=top_k)
                answer_text = gen.answer
                err = None
            except Exception as e:
                answer_text = ""
                err = repr(e)
            gen_ms = int((time.perf_counter() - t0) * 1000)
        else:
            answer_text = ""
            gen_ms = 0
            err = None

        answer_eval = _evaluate_answer(item, answer_text) if include_llm else {}

        results.append({
            "id": item["id"],
            "question": q,
            "retrieval": retrieval_eval,
            "answer": answer_text,
            "answer_eval": answer_eval,
            "latency_ms": gen_ms,
            "error": err,
        })

        flag_recall = "Y" if retrieval_eval["recall_at_4"] else "-"
        if include_llm:
            met = answer_eval.get("must_contain_met", True)
            ref = answer_eval.get("refused", False)
            forb = answer_eval.get("must_not_contain_met", True)
            flag_ans = "Y" if met else "MISS"
            flag_ref = "Y" if ref else "n"
            flag_forb = "OK" if forb else "VIOL"
        else:
            flag_ans = "-"; flag_ref = "-"; flag_forb = "-"
        q_disp = q if len(q) <= 60 else q[:57] + "..."
        print(f"{item['id']:22} {flag_recall:10} {flag_ans:12} {flag_ref:6} {flag_forb:6}  {q_disp}")

    # ---- summary ----
    print("\nSummary")
    print("-------")
    n_recall = sum(1 for r in results if r["retrieval"]["recall_at_4"])
    print(f"Retrieval Recall@4: {n_recall}/{len(in_scope)} "
          f"= {n_recall / max(1, len(in_scope)) * 100:.1f}%  (in-scope items)")

    if include_llm:
        n_must = sum(1 for r in results if r["id"].startswith("in-scope") and r["answer_eval"].get("must_contain_met"))
        print(f"Answer presence   : {n_must}/{len(in_scope)} "
              f"= {n_must / max(1, len(in_scope)) * 100:.1f}%  (in-scope items)")

        n_forb = sum(1 for r in results if r["answer_eval"].get("must_not_contain_met"))
        print(f"Forbidden absent  : {n_forb}/{len(items)} "
              f"= {n_forb / len(items) * 100:.1f}%  (all items)")

        n_refusals_correct = sum(
            1 for r in results
            if r["id"].startswith("out-of-scope") and r["answer_eval"].get("refused")
        )
        print(f"Refusal accuracy  : {n_refusals_correct}/{len(out_of_scope)} "
              f"= {n_refusals_correct / max(1, len(out_of_scope)) * 100:.1f}%  (out-of-scope items)")

        n_no_hallucination_in_scope = sum(
            1 for r in results
            if r["id"].startswith("in-scope")
            and not r["answer_eval"].get("refused")
            and r["answer_eval"].get("must_contain_met")
        )
        print(f"End-to-end in-scope: {n_no_hallucination_in_scope}/{len(in_scope)} "
              f"= {n_no_hallucination_in_scope / max(1, len(in_scope)) * 100:.1f}%")

    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results written to {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="Skip LLM generation; measure retrieval only")
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()
    main(include_llm=not args.no_llm, top_k=args.top_k)