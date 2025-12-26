import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings

from src.settings import (
    DATA_PDF_PATH,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def main():
    # 1. Load the Bible PDF
    if not os.path.exists(DATA_PDF_PATH):
        raise FileNotFoundError(f"Bible PDF not found in {DATA_PDF_PATH}")

    print(f" Loading the Bible from: {DATA_PDF_PATH}")
    loader = PyPDFLoader(DATA_PDF_PATH)
    pages = loader.load()  # each page becomes a Document object

    # 2. Split into chunks
    print(" Splitting the Bible into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(pages)

    # Add metadata to each chunk
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source"] = "bible.pdf"

    print(f"Total chunks created: {len(chunks)}")

    # 3. Create embeddings
    embeddings = get_embeddings()

    # 4. Build the vector store (vector database)
    print("Creating Chroma vector store...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    vectordb.persist() #write all vector database file permanently
    print(f"Chroma vector store saved to: {CHROMA_DIR}")

if __name__ == "__main__":
    main()
