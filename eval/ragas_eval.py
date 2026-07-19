"""RAGAS evaluation of the Bible RAG assistant (LLM-as-judge metrics).

Complements the deterministic keyword harness in eval/evaluate.py with the
industry-standard RAGAS metrics, which judge quality with an LLM instead of
keyword overlap:

  * faithfulness        - is every claim in the answer supported by the
                          retrieved passages? (the grounding guarantee)
  * answer_relevancy    - does the answer actually address the question?
  * context_precision   - are the retrieved passages relevant / well-ranked?
  * context_recall      - did retrieval cover what the reference answer needs?

Needs an OpenAI key and a built vector DB. RAGAS deps are heavy and eval-only;
install them separately (see eval/requirements-eval.txt), NOT in the app image:
    pip install -r eval/requirements-eval.txt
    python -m eval.ragas_eval
Writes eval/ragas_results.csv and eval/ragas_results.json.
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)

from src.rag_qa import answer_with_sources
from src.settings import OPENAI_MODEL

HERE = os.path.dirname(__file__)
if len(sys.argv) > 1:
    DATASET_PATH = sys.argv[1]
    _base = os.path.splitext(DATASET_PATH)[0] + "_results"
else:
    DATASET_PATH = os.path.join(HERE, "ragas_dataset.json")
    _base = os.path.join(HERE, "ragas_results")
CSV_PATH = _base + ".csv"
JSON_PATH = _base + ".json"


def build_samples():
    """Run the RAG pipeline over the ground-truth set and package RAGAS samples."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        cases = json.load(f)["cases"]

    samples = []
    for case in cases:
        question = case["question"]
        answer, sources = answer_with_sources(question)
        contexts = [s["text"] for s in sources] or ["(no context retrieved)"]
        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=case["reference"],
        ))
    return samples


def main():
    print("Running RAG pipeline over the ground-truth set...")
    samples = build_samples()
    dataset = EvaluationDataset(samples=samples)

    # The evaluator (judge) model is independent of the app's answering model.
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=OPENAI_MODEL, temperature=0))
    evaluator_emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]

    print(f"Scoring {len(samples)} samples with RAGAS (this makes several LLM calls)...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_emb,
    )

    df = result.to_pandas()
    df.to_csv(CSV_PATH, index=False)

    # Average the numeric metric columns (ignore the text input columns).
    input_cols = {"user_input", "retrieved_contexts", "response", "reference"}
    metric_cols = [c for c in df.columns if c not in input_cols]
    summary = {c: round(float(df[c].mean()), 4) for c in metric_cols}

    print("\n=== RAGAS SUMMARY (mean over samples) ===")
    for name, score in summary.items():
        print(f"{name:35}: {score}")

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "n_samples": len(samples)}, f, indent=2)
    print(f"\nWrote {CSV_PATH} and {JSON_PATH}")

    return summary


if __name__ == "__main__":
    main()
