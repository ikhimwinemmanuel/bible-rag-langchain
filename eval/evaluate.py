"""Evaluation harness for the Bible RAG assistant.

Measures three things against a labeled dataset (eval/dataset.json):

  1. Retrieval recall@k  - for answerable questions, did retrieval surface a
     chunk containing the expected keyword(s)? (retrieval quality)
  2. Refusal accuracy    - did the system answer answerable questions and
     refuse off-topic ones? (the project's headline grounding guarantee)
  3. Answer coverage     - for answered questions, did the final answer contain
     the expected keyword(s)? (end-to-end correctness)

Run from the project root (needs a built vector DB and an OpenAI key):
    python -m eval.evaluate                     # default: eval/dataset.json
    python -m eval.evaluate eval/dataset_hard.json

Writes a machine-readable summary next to the dataset (results.json for the
default set, <name>_results.json for any other).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_qa import (
    answer_question,
    load_vectordb,
    retrieve_context,
    format_context,
    NO_CONTEXT_MESSAGE,
)

HERE = os.path.dirname(__file__)
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
    RESULTS_PATH = os.path.splitext(DATASET_PATH)[0] + "_results.json"
else:
    DATASET_PATH = os.path.join(HERE, "dataset.json")
    RESULTS_PATH = os.path.join(HERE, "results.json")


def contains_any(text, keywords):
    text_low = text.lower()
    return any(kw.lower() in text_low for kw in keywords)


def _norm(text):
    """Normalize curly apostrophes/quotes so refusal detection is punctuation-proof."""
    return text.strip().replace("’", "'").replace("‘", "'")


def is_refusal(answer):
    return _norm(answer) == _norm(NO_CONTEXT_MESSAGE)


def evaluate():
    with open(DATASET_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    vectordb = load_vectordb()

    # Counters
    retrieval_hits = retrieval_total = 0
    refusal_correct = 0
    coverage_hits = coverage_total = 0
    rows = []

    for case in cases:
        q = case["question"]
        answerable = case["answerable"]
        keywords = case["expected_keywords"]

        # --- Retrieval quality (answerable cases only) ---
        retrieval_hit = None
        if answerable and keywords:
            context = format_context(retrieve_context(vectordb, q))
            retrieval_hit = contains_any(context, keywords)
            retrieval_total += 1
            retrieval_hits += int(retrieval_hit)

        # --- End-to-end answer ---
        answer = answer_question(q)
        refused = is_refusal(answer)

        # --- Refusal accuracy ---
        refusal_ok = (not refused) if answerable else refused
        refusal_correct += int(refusal_ok)

        # --- Answer coverage (answered + answerable) ---
        coverage_hit = None
        if answerable and keywords and not refused:
            coverage_total += 1
            coverage_hit = contains_any(answer, keywords)
            coverage_hits += int(coverage_hit)

        rows.append({
            "question": q,
            "answerable": answerable,
            "refused": refused,
            "refusal_ok": refusal_ok,
            "retrieval_hit": retrieval_hit,
            "coverage_hit": coverage_hit,
        })

    def pct(n, d):
        return round(100.0 * n / d, 1) if d else None

    summary = {
        "n_cases": len(cases),
        "retrieval_recall_at_k_pct": pct(retrieval_hits, retrieval_total),
        "refusal_accuracy_pct": pct(refusal_correct, len(cases)),
        "answer_coverage_pct": pct(coverage_hits, coverage_total),
        "counts": {
            "retrieval": f"{retrieval_hits}/{retrieval_total}",
            "refusal": f"{refusal_correct}/{len(cases)}",
            "coverage": f"{coverage_hits}/{coverage_total}",
        },
    }

    # --- Report ---
    print(f"\n{'Q':60} {'ans?':5} {'refuse_ok':10} {'retr':5} {'cov':5}")
    print("-" * 90)
    for r in rows:
        def mark(v):
            return "-" if v is None else ("Y" if v else "N")
        print(f"{r['question'][:60]:60} "
              f"{str(r['answerable']):5} "
              f"{('OK' if r['refusal_ok'] else 'FAIL'):10} "
              f"{mark(r['retrieval_hit']):5} "
              f"{mark(r['coverage_hit']):5}")

    print("\n=== SUMMARY ===")
    print(f"Retrieval recall@k : {summary['retrieval_recall_at_k_pct']}%  ({summary['counts']['retrieval']})")
    print(f"Refusal accuracy   : {summary['refusal_accuracy_pct']}%  ({summary['counts']['refusal']})")
    print(f"Answer coverage    : {summary['answer_coverage_pct']}%  ({summary['counts']['coverage']})")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"\nWrote {RESULTS_PATH}")

    return summary


if __name__ == "__main__":
    evaluate()
