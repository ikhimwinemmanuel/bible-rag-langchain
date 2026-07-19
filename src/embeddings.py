from langchain_openai import OpenAIEmbeddings

from src.settings import EMBEDDINGS_BACKEND, HF_EMBEDDINGS_MODEL


def get_embeddings():
    """Return the embedding model for the configured backend."""
    if EMBEDDINGS_BACKEND.lower() == "openai":
        return OpenAIEmbeddings()

    # HuggingFace backend is optional. Import lazily so OpenAI users don't need
    # sentence-transformers / torch installed, and prefer the maintained
    # langchain-huggingface package over the deprecated community import.
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)
