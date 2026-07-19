import os

from langchain_core.documents import Document
from langchain_chroma import Chroma

from src.embeddings import get_embeddings
from src.bible_parser import parse_verses
from src.settings import (
    DATA_PDF_PATH,
    CHROMA_DIR,
    CHUNK_SIZE,
)


def build_chunks(verses):
    """Group consecutive verses into ~CHUNK_SIZE chunks that never cross a chapter.

    Keeping each chunk inside a single chapter means every chunk carries a clean,
    human-readable citation like "Genesis 1:1-5" instead of an opaque chunk id.
    """
    chunks = []
    current = []
    current_len = 0
    prev_key = None

    def flush():
        nonlocal current, current_len
        if not current:
            return
        first, last = current[0], current[-1]
        book, chapter = first["book"], first["chapter"]
        if first["verse"] == last["verse"]:
            ref = f"{book} {chapter}:{first['verse']}"
        else:
            ref = f"{book} {chapter}:{first['verse']}-{last['verse']}"
        # Prefix each verse with its number so the LLM can cite specific verses.
        text = " ".join(f"{v['verse']} {v['text']}" for v in current)
        chunks.append(Document(
            page_content=text,
            metadata={
                "book": book,
                "chapter": chapter,
                "verse_start": first["verse"],
                "verse_end": last["verse"],
                "ref": ref,
                "source": "bible.pdf",
            },
        ))
        current = []
        current_len = 0

    for v in verses:
        key = (v["book"], v["chapter"])
        if prev_key is not None and key != prev_key:
            flush()  # never let a chunk span two chapters
        current.append(v)
        current_len += len(v["text"])
        if current_len >= CHUNK_SIZE:
            flush()
        prev_key = key
    flush()

    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
    return chunks


def main():
    # 1. Load and parse the Bible PDF into structured verses
    if not os.path.exists(DATA_PDF_PATH):
        raise FileNotFoundError(f"Bible PDF not found in {DATA_PDF_PATH}")

    print(f"Parsing the Bible from: {DATA_PDF_PATH}")
    verses = list(parse_verses(DATA_PDF_PATH))
    print(f"Parsed {len(verses)} verses")

    # 2. Group verses into citation-friendly chunks
    print("Building verse-range chunks...")
    chunks = build_chunks(verses)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Create embeddings
    embeddings = get_embeddings()

    # 4. Build the vector store. Cosine distance keeps relevance scores in [0, 1]
    # so SCORE_THRESHOLD in retrieval is a meaningful "minimum similarity" gate.
    print("Creating Chroma vector store...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Chroma with a persist_directory writes to disk automatically.
    print(f"Chroma vector store saved to: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
