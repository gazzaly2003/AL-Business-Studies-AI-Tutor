"""
app.py
------
A local AI chat website (runs on your own PC) where a student can type
a Business Studies question and get an answer grounded in the actual
syllabus content you loaded with ingest.py.

Run with:
    streamlit run app.py
"""

import requests
import streamlit as st
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import Chroma

DB_DIR = "chroma_db"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"   # pull with: ollama pull llama3.2:3b
OLLAMA_URL = "http://127.0.0.1:11434"


class SimpleOllamaEmbeddings(Embeddings):
    """Talks directly to Ollama's stable /api/embeddings endpoint, one
    text at a time, so the question is embedded the same way the
    documents were in ingest.py."""

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


PROMPT_TEMPLATE = """You are a patient, knowledgeable tutor for the Sri Lankan
Advanced Level Business Studies syllabus.

Use ONLY the context below to answer the student's question. Explain it the
way a good teacher would: clear definitions, short examples, and exam-style
structure where useful. If the context does not contain the answer, say
clearly that this isn't covered in the material you have, instead of guessing.

Context:
{context}

Student's question: {question}

Answer:"""

st.set_page_config(
    page_title="Business Studies AI Tutor — by Gazzaly",
    page_icon="📘",
    layout="centered",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; max-width: 780px; }
        .app-header { text-align: center; margin-bottom: 0.2rem; }
        .app-header h1 { font-size: 1.9rem; margin-bottom: 0.1rem; }
        .app-credit {
            text-align: center;
            color: #8a8f98;
            font-size: 0.95rem;
            margin-bottom: 1.6rem;
        }
        .app-credit b { color: #4f8cff; }
        [data-testid="stChatMessage"] { border-radius: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-header"><h1>📘 Business Studies AI Tutor</h1></div>'
    '<div class="app-credit">Created by <b>Gazzaly</b> · '
    'A/L Business Studies (Sri Lanka), English medium</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "Ask any question or doubt from the A/L Business Studies syllabus. "
        "Answers are generated only from the syllabus material loaded into "
        "this app, with sources shown so you can double-check every answer."
    )
    st.markdown("---")
    st.caption("Runs fully offline on a local AI model (Ollama).")
    st.caption("Built by **Gazzaly** as an AI/ML portfolio project.")


@st.cache_resource(show_spinner=False)
def load_pipeline():
    embeddings = SimpleOllamaEmbeddings()
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    llm = OllamaLLM(model=LLM_MODEL)
    return db, llm


try:
    db, llm = load_pipeline()
except Exception as e:
    st.error(
        "Couldn't load the database. Make sure you've run `python ingest.py` "
        "first, and that Ollama is running in the background.\n\n"
        f"Details: {e}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Where this came from"):
                for source, page in msg["sources"]:
                    st.write(f"- **{source}**, page {page}")

question = st.chat_input("Ask a Business Studies question...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your syllabus material..."):
            docs = db.similarity_search(question, k=4)
            context = "\n\n".join(doc.page_content for doc in docs)
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt)

        st.markdown(answer)

        sources = [
            (doc.metadata.get("source", "unknown file"), doc.metadata.get("page", "?"))
            for doc in docs
        ]
        with st.expander("📄 Where this came from"):
            for source, page in sources:
                st.write(f"- **{source}**, page {page}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
