
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from src.settings import EMBEDDINGS_BACKEND, HF_EMBEDDINGS_MODEL

def get_embeddings():
    """Return the correct embedding model depending on backend."""
    if EMBEDDINGS_BACKEND.lower() == "openai":
        return OpenAIEmbeddings()
    else:
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDINGS_MODEL)
