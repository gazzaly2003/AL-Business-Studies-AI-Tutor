"""
ingest_cloud.py
----------------
Same job as ingest.py, but built for the CLOUD-deployed version of the
assistant: it uses a small, free, self-contained embedding model
(sentence-transformers) instead of Ollama, so it works on Streamlit
Community Cloud where you can't run your own background AI server.

Run this ONCE on your own PC before deploying (and again if you add
new PDFs), then push the resulting chroma_db_cloud/ folder to GitHub
along with your code:

    python ingest_cloud.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
DB_DIR = "chroma_db_cloud"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_all_pdfs(folder):
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
        print(f"No files found in '{DATA_DIR}/'. Add your syllabus PDFs "
              f"there first, then re-run this script.")
        return

    print("Step 1/3 — Loading PDFs...")
    docs = load_all_pdfs(DATA_DIR)
    print(f"  Loaded {len(docs)} pages total.")

    print("Step 2/3 — Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"  Created {len(chunks)} chunks.")

    print("Step 3/3 — Creating embeddings and saving to ChromaDB "
          "(downloads a small model the first time, then runs locally)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)

    print(f"\nDone. Cloud vector database saved to '{DB_DIR}/'.")
    print("Commit this folder to your GitHub repo along with app_cloud.py "
          "before deploying to Streamlit Community Cloud.")


if __name__ == "__main__":
    main()
