import functools
import re

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.embeddings import get_embeddings
from src.settings import (
    CHROMA_DIR,
    OPENAI_MODEL,
    TOP_K,
    FINAL_K,
    SCORE_THRESHOLD,
)
from src.prompts import SYSTEM_PROMPT, BIBLE_RAG_PROMPT, NO_CONTEXT_MESSAGE

# Canned reply for greetings, kept in sync with SYSTEM_PROMPT. The refusal
# message (NO_CONTEXT_MESSAGE) is imported so the gate and the LLM agree exactly.
GREETING_REPLY = (
    "Hello. I am a RAG-based Bible assistant. "
    "How can I help you with a Bible-related question?"
)

# Short inputs treated as greetings so we skip retrieval and an LLM call entirely.
GREETINGS = {
    "hi", "hello", "hey", "hiya", "hola", "greetings",
    "good morning", "good afternoon", "good evening",
}


@functools.lru_cache(maxsize=1)
def load_vectordb():
    """Load the existing Chroma vector database once and reuse it."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,  # MUST match ingestion embeddings
    )


@functools.lru_cache(maxsize=1)
def get_llm():
    """Instantiate the chat model once; temperature=0 keeps answers grounded."""
    return ChatOpenAI(model=OPENAI_MODEL, temperature=0)


def is_greeting(text):
    """Return True for bare greetings so we can reply without retrieval."""
    normalized = text.strip().lower().strip("!.? ")
    return normalized in GREETINGS


def _bm25_preprocess(text):
    """Lowercase word-token tokenizer so keyword matching is case-insensitive."""
    return re.findall(r"\w+", text.lower())


@functools.lru_cache(maxsize=1)
def get_bm25_retriever():
    """Build a BM25 keyword retriever over the same chunks as the vector store.

    Dense (embedding) search can miss the exact verse a specific-detail question
    needs when its signal is diluted in a verse-range chunk; BM25 recovers those
    by matching rare surface terms (names, numbers, "MENE", "Euphrates", ...).
    """
    vectordb = load_vectordb()
    data = vectordb.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(data["documents"], data["metadatas"])
    ]
    retriever = BM25Retriever.from_documents(docs, preprocess_func=_bm25_preprocess)
    retriever.k = TOP_K
    return retriever


def _rrf_fuse(ranked_lists, top_n, k=60):
    """Reciprocal Rank Fusion: merge ranked doc lists into one, deduped by ref."""
    scores = {}
    docs_by_key = {}
    for docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = doc.metadata.get("ref") or doc.page_content[:64]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            docs_by_key[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ranked[:top_n]]


def retrieve_context(vectordb, question):
    """Hybrid retrieval with a dense grounding gate.

    The refuse/answer decision uses the dense cosine relevance score (a good
    off-topic detector: SCORE_THRESHOLD separates on- from off-topic). Only if
    that gate passes do we fuse the dense and BM25 candidate lists with RRF, so
    off-topic questions can never be answered via a stray keyword match, while
    on-topic answers gain BM25's exact-term recall.
    """
    dense_scored = vectordb.similarity_search_with_relevance_scores(question, k=TOP_K)

    # Grounding gate: nothing relevant enough -> refuse (return no context).
    if not dense_scored or max(score for _, score in dense_scored) < SCORE_THRESHOLD:
        return []

    dense_docs = [doc for doc, _ in dense_scored]
    bm25_docs = get_bm25_retriever().invoke(question)
    return _rrf_fuse([dense_docs, bm25_docs], top_n=FINAL_K)


def format_context(docs):
    """Combine retrieved chunks into a single string, tagged by verse reference."""
    combined = ""
    for d in docs:
        ref = d.metadata.get("ref") or f"Chunk {d.metadata.get('chunk_id', 'unknown')}"
        combined += f"[{ref}] {d.page_content}\n\n"
    return combined


def generate_answer(question, context):
    """Send the question + retrieved context to the LLM."""
    llm = get_llm()
    prompt = BIBLE_RAG_PROMPT.format(question=question, context=context)

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return response.content


def answer_with_sources(question):
    """Answer a question and return (answer_text, sources).

    sources is a list of {"ref", "text"} dicts for the passages that grounded
    the answer — empty for greetings, empty input, or refusals.
    """
    question = (question or "").strip()

    if not question:
        return "Please ask a Bible-related question.", []

    if is_greeting(question):
        return GREETING_REPLY, []

    vectordb = load_vectordb()
    docs = retrieve_context(vectordb, question)

    # Grounding guarantee: no relevant passages -> refuse instead of guessing.
    if not docs:
        return NO_CONTEXT_MESSAGE, []

    context = format_context(docs)
    answer = generate_answer(question, context)
    sources = [
        {"ref": d.metadata.get("ref", "unknown"), "text": d.page_content}
        for d in docs
    ]
    return answer, sources


def answer_question(question):
    """Backward-compatible wrapper returning just the answer text."""
    answer, _ = answer_with_sources(question)
    return answer
