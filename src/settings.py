import os
from dotenv import load_dotenv

# Load environment variables from .env file when the project runs
load_dotenv()

# model and embedding setting

# Which LLM to use for answering questions
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Which embedding backend to use: "openai" or "huggingface"
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "openai")

# If using HuggingFace embeddings, this is the model name
HF_EMBEDDINGS_MODEL = os.getenv(
    "HF_EMBEDDINGS_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
) 

# ----------- CHUNKING SETTINGS -----------

# Number of characters per chunk
CHUNK_SIZE = 1000

# Overlap between chunks to avoid splitting sentences unnaturally
CHUNK_OVERLAP = 150

# ----------- RETRIEVAL SETTINGS -----------

# How many chunks to retrieve for each question
TOP_K = 5

# Minimum relevance score (0=unrelated, 1=identical) for a chunk to be accepted.
# Backend-specific: with OpenAI embeddings + verse-range chunks, the eval set
# shows off-topic queries top out at ~0.78 and on-topic bottom out at ~0.81,
# so 0.80 separates them cleanly. HuggingFace embeddings have a different
# distribution — retune if you switch EMBEDDINGS_BACKEND. Override at runtime
# with SCORE_THRESHOLD.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.80"))

# ----------- PATHS -----------

# Path to your Bible PDF
DATA_PDF_PATH = os.path.join("data", "bible.pdf")

# Where Chroma will store your vector database
CHROMA_DIR = "db"
