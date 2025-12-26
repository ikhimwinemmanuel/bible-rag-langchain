import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from src.embeddings import get_embeddings

from src.settings import (
    CHROMA_DIR,
    OPENAI_MODEL,
    TOP_K,
    SCORE_THRESHOLD,
)
from src.prompts import SYSTEM_PROMPT, BIBLE_RAG_PROMPT

def load_vectordb():
    """Load the existing Chroma vector database."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings  # MUST match ingestion embeddings
    )

def retrieve_context(vectordb, question):
    docs_and_scores = vectordb.similarity_search_with_score(
        question,
        k=TOP_K
    )
    return [doc for doc, _ in docs_and_scores]


def format_context(docs):
    """Combine text from retrieved chunks into a single string."""
    combined = ""
    for d in docs:
        chunk_id = d.metadata.get("chunk_id", "unknown")
        combined += f"[Chunk {chunk_id}] {d.page_content}\n\n"
    return combined

def generate_answer(question, context):
    """Send the question + context to the LLM."""
    llm = ChatOpenAI(model=OPENAI_MODEL)
    prompt = BIBLE_RAG_PROMPT.format(question=question, context=context)

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ])
    return response.content

def answer_question(question):
    question = question.strip().lower()
    vectordb = load_vectordb()
    docs = retrieve_context(vectordb, question)
    context = format_context(docs)
    return generate_answer(question, context)
