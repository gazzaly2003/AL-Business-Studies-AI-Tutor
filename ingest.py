"""
ingest.py
---------
Reads every PDF inside the /data folder (syllabus, teacher's guide,
past papers), splits them into small overlapping chunks, converts each
chunk into a vector using a LOCAL embedding model (nothing is sent to
the internet), and saves everything into a local ChromaDB database.

Run this ONCE (and again any time you add new PDFs) before starting
the chat app:

    python ingest.py
"""

import os
import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"   # pull with: ollama pull nomic-embed-text
OLLAMA_URL = "http://127.0.0.1:11434"


class SimpleOllamaEmbeddings(Embeddings):
    """
    Talks directly to Ollama's older, stable /api/embeddings endpoint
    (one text at a time) instead of going through the ollama Python
    package, which has a buggy batch-tokenization path on some
    versions that tries to hit a random local port and fails.
    """

    def __init__(self, model=EMBED_MODEL, base_url=OLLAMA_URL):
        self.model = model
        self.base_url = base_url

    def embed_documents(self, texts):
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text):
        return self._embed_one(text)

    def _embed_one(self, text):
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]


def load_all_pdfs(folder):
    """Load every .pdf file in the folder (and subfolders) into documents."""
    documents = []
    for root, _, files in os.walk(folder):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                path = os.path.join(root, filename)
                print(f"  Reading: {path}")
                loader = PyPDFLoader(path)
                documents.extend(loader.load())
    return documents


def main():
    if not os.path.isdir(DATA_DIR) or not os.listdir(DATA_DIR):
        print(f"No files found in '{DATA_DIR}/'. Add your syllabus, teacher's "
              f"guide, and past-paper PDFs there first, then re-run this script.")
        return

    print("Step 1/3 — Loading PDFs...")
    docs = load_all_pdfs(DATA_DIR)
    print(f"  Loaded {len(docs)} pages total.")

    print("Step 2/3 — Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks.")

    print("Step 3/3 — Creating embeddings and saving to ChromaDB "
          "(this can take a few minutes the first time, since each "
          "chunk is processed one at a time)...")
    embeddings = SimpleOllamaEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

    print(f"\nDone. Vector database saved to '{DB_DIR}/'.")
    print("You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
