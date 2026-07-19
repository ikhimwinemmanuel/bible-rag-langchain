import functools

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

# Canned replies for the two non-retrieval paths, kept in sync with SYSTEM_PROMPT.
GREETING_REPLY = (
    "Hello. I am a RAG-based Bible assistant. "
    "How can I help you with a Bible-related question?"
)
NO_CONTEXT_MESSAGE = "I couldn't find a grounded answer in the retrieved passages."

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


def retrieve_context(vectordb, question):
    """Return only the chunks whose relevance score clears SCORE_THRESHOLD.

    Uses relevance scores (0=unrelated, 1=identical) rather than raw distance,
    so SCORE_THRESHOLD is a meaningful "minimum similarity" gate. This is what
    lets the assistant refuse to answer when nothing relevant is retrieved.
    """
    docs_and_scores = vectordb.similarity_search_with_relevance_scores(
        question,
        k=TOP_K,
    )
    return [doc for doc, score in docs_and_scores if score >= SCORE_THRESHOLD]


def format_context(docs):
    """Combine text from retrieved chunks into a single string."""
    combined = ""
    for d in docs:
        chunk_id = d.metadata.get("chunk_id", "unknown")
        combined += f"[Chunk {chunk_id}] {d.page_content}\n\n"
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


def answer_question(question):
    question = (question or "").strip()

    if not question:
        return "Please ask a Bible-related question."

    if is_greeting(question):
        return GREETING_REPLY

    vectordb = load_vectordb()
    docs = retrieve_context(vectordb, question)

    # Grounding guarantee: no relevant passages -> refuse instead of guessing.
    if not docs:
        return NO_CONTEXT_MESSAGE

    context = format_context(docs)
    return generate_answer(question, context)
