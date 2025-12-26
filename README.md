## Bible RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers questions strictly from Bible text, refusing to guess when evidence is missing.

Built as a practical exploration of RAG system design, semantic search in vector databases, and System prompt design in LLM usage.

## Overview

This project implements an end-to-end RAG pipeline using the Bible as a fixed knowledge source.
Instead of relying on a model’s memory, each answer is generated only after retrieving relevant passages from the source text.

If no supporting passages are found, the system explicitly responds that it cannot produce a grounded answer.

## Tech Stack

LangChain – RAG orchestration

ChromaDB – vector database for semantic search

OpenAI / Hugging Face – interchangeable embedding backends

Gradio – lightweight web interface

Hugging Face Spaces – deployment

## Architecture Summary

Ingestion

Load Bible PDF

Split text into overlapping chunks

Generate embeddings

Store vectors in ChromaDB

Retrieval & Answering

Retrieve top-K most relevant chunks per question

Assemble retrieved context

Generate answers using an LLM

Enforce grounding and citations strictly from Bible text

Prompting

Context-only responses

No hallucination

Chunk and verse references included

## Project Structure

```text
bible-rag-langchain/
├── app.py                     # Gradio application entry point
├── data/
│   └── bible.pdf              # Bible source document
├── src/
│   ├── ingest.py              # PDF ingestion + vector store creation
│   ├── rag_qa.py              # Retrieval and QA pipeline
│   ├── embeddings.py          # Embedding backend abstraction
│   ├── prompts.py             # System and user prompt templates
│   └── settings.py            # Configuration and constants
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Local containerization (optional)
├── .dockerignore              # Docker ignore rules
└── README.md                  # Project documentation
```

## Configuration

Environment variables:

OPENAI_API_KEY=your_api_key
EMBEDDINGS_BACKEND=openai   # or huggingface

Secrets are managed via Hugging Face Spaces during deployment.

## Deployment

The application is deployed on Hugging Face Spaces using Gradio.
Docker was used locally to validate portability and cross-platform deployment readiness.

Notes:
Answers are generated only from retrieved Bible passages
The architecture allows easy switching between paid and free embedding models

## Live Demo

Try the live application on Hugging Face Spaces:

https://huggingface.co/spaces/Ikhimwin/bible-rag-langchain 

