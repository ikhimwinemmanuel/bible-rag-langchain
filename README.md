---
title: Bible RAG Assistant
emoji: 📖
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: "3.12"
pinned: false
---

# Bible RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions **strictly from Bible text**, cites the exact verses it used, and **refuses to answer when no relevant passage is retrieved** — rather than guessing.

Built as a practical exploration of RAG system design: semantic retrieval, grounding guarantees, verse-level citations, and measured evaluation.

## Why this project is more than a demo

Most RAG tutorials stop at "retrieve top-k, then answer." This project adds the parts that make a RAG system trustworthy and measurable:

- **Enforced grounding.** Off-topic questions ("What is the capital of France?") are refused *before* an LLM call, using a measured relevance threshold. The refusal is a real gate, not just a prompt instruction.
- **Verse-level citations.** The KJV PDF is parsed into 31,102 structured verses, so answers cite `[Genesis 1:3-5]` — not an opaque chunk id — and the UI shows the source passages.
- **An evaluation harness.** A labeled dataset and scoring script report retrieval recall, refusal accuracy, and answer coverage, so quality is a number, not a vibe.

## Results

The project is evaluated two ways: a fast deterministic harness for CI-style checks, and RAGAS (LLM-as-judge) for deeper quality metrics.

**1. Custom harness** (`eval/dataset.json`, 20 cases; `python -m eval.evaluate`):

| Metric | Score | Meaning |
|---|---|---|
| Retrieval recall@k | 100% (15/15) | Relevant passage retrieved for answerable questions |
| Refusal accuracy | 100% (20/20) | Answered in-scope questions, refused off-topic ones |
| Answer coverage | 100% (15/15) | Final answer contained the expected fact |

On the harder set (`eval/dataset_hard.json`, 24 cases incl. near-miss off-topic), the same harness also scores 100% / 100% / 100% — natural-language questions retrieve well, and the LLM refuses near-miss questions ("What year was Jesus born?") as a second layer even when they pass the score gate.

**2. RAGAS** (LLM-as-judge; `python -m eval.ragas_eval [dataset]`).

The **hard set** (`eval/ragas_dataset_hard.json`) stresses the system with precise-detail questions (ages/numbers), non-KJV paraphrasing, and precision traps. It exposed a retrieval weakness that motivated **hybrid retrieval** (see below); the headline result is the context-recall gain:

| RAGAS metric (hard set) | Dense only | Hybrid (dense + BM25) |
|---|---|---|
| **Context recall** | **0.87** | **0.93** |
| Context precision | 0.83 | 0.81 |
| Faithfulness | ~0.93 | ~0.95 |
| Answer relevancy | ~0.98 | ~0.98 |

**How to read these numbers honestly:**
- **Context recall (0.87 → 0.93) is the stable, real win.** Retrieval is deterministic, so this figure is reproducible run-to-run. It is corroborated by concrete cases: verses dense search missed entirely — Lot's wife (Gen 19:26), the writing on Belshazzar's wall (Dan 5:25), the rivers of Eden (Gen 2:14) — are now retrieved via BM25 keyword matching.
- **Trade-off:** easy-set context precision drops (~0.80 → ~0.67) because BM25 adds keyword candidates the LLM then ignores; answer-quality metrics do not suffer.
- **Faithfulness/relevancy are reported with `~`** because, on 15-case sets, the RAGAS LLM judge varies by ±0.1 between identical runs. The small eval set is itself a known limitation — treat these as directional, not precise.

Datasets are intentionally small and readable; extend them for tighter confidence intervals. RAGAS deps are heavy and eval-only — see `eval/requirements-eval.txt`.

## How retrieval and grounding work

1. **Grounding gate (dense).** Score the top-K chunks with **cosine relevance** (Chroma is built with `hnsw:space: cosine` so scores land in `[0, 1]`). If the best score is below `SCORE_THRESHOLD`, return a fixed refusal message — **no LLM call**. The gate is deliberately dense-only: cosine relevance is a good off-topic detector, whereas BM25 scores are not, so a stray keyword can never let an off-topic question through.
2. **Hybrid retrieval (dense + BM25).** Once the gate passes, fuse the dense candidates with a BM25 keyword retriever over the same chunks using **Reciprocal Rank Fusion**, keeping the top `FINAL_K`. BM25 recovers the exact verse a specific-detail question needs (names, numbers, rare terms) when the embedding signal is diluted across a verse-range chunk.
3. **Answer.** Pass the fused passages (labelled with their verse references) to the LLM under a grounding system prompt; it cites the references it used or emits the refusal message.

The threshold is backend-specific and was chosen from measured data: with OpenAI embeddings and verse-range chunks, off-topic queries top out around 0.78 while on-topic queries bottom out around 0.81, so the default `SCORE_THRESHOLD = 0.80` separates them cleanly. `FINAL_K` (8) exceeds `TOP_K` (5) so BM25's finds are *added* to the dense results rather than evicting them. Both are overridable via environment / settings.

## Tech Stack

- **LangChain** – RAG orchestration
- **ChromaDB** – vector database for semantic search (cosine distance)
- **BM25 (rank-bm25)** – sparse keyword retriever for hybrid retrieval
- **OpenAI / Hugging Face** – interchangeable embedding backends
- **Gradio** – web interface
- **Hugging Face Spaces** – deployment

## Architecture

**Ingestion** (`src/bible_parser.py`, `src/ingest.py`)
- Parse the KJV PDF into structured verses (book, chapter, verse) using the chapter headers and verse-number markers in the source
- Group consecutive verses into ~1000-char chunks that never cross a chapter, so each chunk carries a clean citation reference
- Embed and store in Chroma with a cosine distance metric

**Retrieval & answering** (`src/rag_qa.py`)
- Dense grounding gate + hybrid (dense + BM25) retrieval fused with RRF (see above)
- Greetings and empty input handled in code (no wasted retrieval / LLM calls)
- Cached vector store, BM25 index, and LLM client
- Answers cite verse references; the UI shows the source passages

## Project Structure

```text
bible-rag-langchain/
├── app.py                     # Gradio application entry point
├── data/
│   └── bible.pdf              # Bible source document (KJV)
├── src/
│   ├── bible_parser.py        # PDF -> structured verses (book/chapter/verse)
│   ├── ingest.py              # Chunking + vector store creation
│   ├── rag_qa.py              # Retrieval, grounding, and QA pipeline
│   ├── embeddings.py          # Embedding backend abstraction
│   ├── prompts.py             # System and user prompt templates
│   └── settings.py            # Configuration and constants
├── eval/
│   ├── dataset.json           # Labeled set for the custom harness
│   ├── evaluate.py            # Retrieval / refusal / coverage metrics
│   ├── ragas_dataset.json     # Ground-truth set for RAGAS
│   ├── ragas_eval.py          # RAGAS LLM-as-judge metrics
│   ├── requirements-eval.txt  # Eval-only deps (RAGAS)
│   └── results.json           # Latest eval results
├── tests/
│   └── sanity_qa.py           # Quick manual smoke test
├── requirements.txt
├── Dockerfile
└── README.md
```

## Configuration

Environment variables (see `.env`):

```bash
OPENAI_API_KEY=your_api_key
EMBEDDINGS_BACKEND=openai        
OPENAI_MODEL=gpt-4o-mini
SCORE_THRESHOLD=0.80             
```

Secrets are managed via Hugging Face Spaces during deployment.

## Running locally

```bash
pip install -r requirements.txt

# 1. Build the vector store from the Bible PDF (writes to db/)
python -m src.ingest

# 2. Launch the app
python app.py            # http://localhost:7860

# Optional: run the evaluation harness or a quick smoke test
python -m eval.evaluate
python -m tests.sanity_qa
```

> Note: Gradio 4.x depends on `pydub`, which needs the `audioop` module removed
> from the Python standard library in 3.13. Use **Python 3.11** (as the
> Dockerfile does) or install the `audioop-lts` backport when running on 3.13.

## Deployment

Deployed on Hugging Face Spaces using Gradio; Docker (`python:3.11-slim`) is used to validate portability. The vector store (`db/`) is gitignored and rebuilt via `python -m src.ingest`.

## Demo

Live application: https://huggingface.co/spaces/Ikhimwin/bible-rag-langchain

![Bible RAG Assistant](assets/demo_ui.webp)

