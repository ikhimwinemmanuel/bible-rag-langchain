"""Manual sanity checks for the RAG pipeline.

Run from the project root so the `src` package resolves:
    python -m tests.sanity_qa
"""
import os
import sys

# Allow running as a plain script (python tests/sanity_qa.py) too.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag_qa import answer_question, GREETING_REPLY, NO_CONTEXT_MESSAGE


def run_sanity_tests():
    # Grounded questions that should retrieve real passages.
    questions = [
        "Where in the Bible does it mention the creation of man?",
        "Who built the ark?",
        "What is the greatest commandment?",
    ]

    # Inputs that should NOT hit the LLM / retrieval at all.
    edge_cases = [
        ("hi", GREETING_REPLY),
        ("What is the capital of France?", NO_CONTEXT_MESSAGE),
    ]

    for q in questions:
        print(f"\nQUESTION: {q}")
        try:
            answer = answer_question(q)
            print(f"ANSWER:\n{answer}")
        except Exception as e:
            print(f"[ERROR] answering question: {e}")

    print("\n--- Edge cases ---")
    for q, expected in edge_cases:
        answer = answer_question(q)
        status = "[PASS]" if answer == expected else "[WARN] (LLM/retrieval path taken)"
        print(f"{status} INPUT: {q!r} -> {answer!r}")


if __name__ == "__main__":
    run_sanity_tests()
