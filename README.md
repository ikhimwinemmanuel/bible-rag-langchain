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

**2. RAGAS** (LLM-as-judge; `python -m eval.ragas_eval [dataset]`):

| RAGAS metric | Easy set (15) | Hard set (15) | Meaning |
|---|---|---|---|
| Faithfulness | 0.93 | 0.93 | Answer claims are supported by the retrieved passages |
| Answer relevancy | 0.90 | 0.98 | Answer actually addresses the question |
| Context precision | 0.80 | 0.83 | Retrieved passages are relevant / well-ranked |
| Context recall | 1.00 | 0.87 | Retrieval covered what the reference answer needs |

The **hard set** (`eval/ragas_dataset_hard.json`) stresses the system with precise-detail questions (ages/numbers), non-KJV paraphrasing, and precision traps. Run any set by passing its path, e.g. `python -m eval.evaluate eval/dataset_hard.json`.

**What the hard set revealed:**
- **Faithfulness holds at 0.93** even as difficulty rises — the system stays grounded (or refuses) rather than hallucinating.
- **Context recall drops to 0.87** — verse-range chunking sometimes fails to surface the exact verse a specific-detail answer needs. Clear next step: hybrid (BM25 + dense) retrieval or finer-grained chunks.
- It also caught a real bug: the retrieval gate and the LLM used two slightly different refusal strings (straight vs. curly apostrophe), making valid refusals undetectable. Fixed by making the refusal message a single source of truth.

Datasets are intentionally small and readable; extend them to stress the system further. RAGAS deps are heavy and eval-only — see `eval/requirements-eval.txt`.

## How grounding works

1. Retrieve the top-K chunks with **cosine relevance scores** (Chroma is built with `hnsw:space: cosine` so scores land in `[0, 1]`).
2. Keep only chunks scoring above `SCORE_THRESHOLD`.
3. If nothing clears the bar, return a fixed refusal message — **no LLM call**.
4. Otherwise, pass the surviving passages (labelled with their verse references) to the LLM under a grounding system prompt.

The threshold is backend-specific and was chosen from measured data: with OpenAI embeddings and verse-range chunks, off-topic queries top out around 0.78 while on-topic queries bottom out around 0.81, so the default `SCORE_THRESHOLD = 0.80` separates them cleanly. It is overridable via the `SCORE_THRESHOLD` environment variable.

## Tech Stack

- **LangChain** – RAG orchestration
- **ChromaDB** – vector database for semantic search (cosine distance)
- **OpenAI / Hugging Face** – interchangeable embedding backends
- **Gradio** – web interface
- **Hugging Face Spaces** – deployment

## Architecture

**Ingestion** (`src/bible_parser.py`, `src/ingest.py`)
- Parse the KJV PDF into structured verses (book, chapter, verse) using the chapter headers and verse-number markers in the source
- Group consecutive verses into ~1000-char chunks that never cross a chapter, so each chunk carries a clean citation reference
- Embed and store in Chroma with a cosine distance metric

**Retrieval & answering** (`src/rag_qa.py`)
- Relevance-scored retrieval with a grounding threshold (see above)
- Greetings and empty input handled in code (no wasted retrieval / LLM calls)
- Cached vector store and LLM client
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
EMBEDDINGS_BACKEND=openai        # or "huggingface"
OPENAI_MODEL=gpt-4o-mini
SCORE_THRESHOLD=0.80             # optional; retune if you change the backend
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

## Write-up

https://medium.com/@ikhimwinemmanuel/building-a-bible-q-a-assistant-with-rag-langchain-chromadb-59543c976199
