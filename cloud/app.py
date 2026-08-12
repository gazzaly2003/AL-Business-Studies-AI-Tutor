"""
app.py (cloud version)
-----------------------
The public-website version of the assistant, meant to be deployed at
cloud/app.py on Streamlit Community Cloud.

Instead of shipping a pre-built vector database (which creates large
files GitHub's uploader rejects), this version builds its knowledge
base automatically from the PDFs in ../data the first time it starts,
using a small free embedding model. Groq's free API generates answers.

Needs a free Groq API key (https://console.groq.com) saved as a
Streamlit secret called GROQ_API_KEY before it will work.
"""

from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# Find the data/ folder regardless of what directory Streamlit runs from
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"

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
    st.caption("Powered by a free cloud AI model (Groq).")
    st.caption("Built by **Gazzaly** as an AI/ML portfolio project.")


@st.cache_resource(show_spinner="Building knowledge base from syllabus PDFs (first load only, this can take a couple of minutes)...")
def load_db():
    pdf_files = list(DATA_DIR.glob("**/*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {DATA_DIR}")

    docs = []
    for pdf_path in pdf_files:
        docs.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return Chroma.from_documents(chunks, embeddings)


try:
    db = load_db()
except Exception as e:
    st.error(f"Couldn't build the knowledge base from PDFs in {DATA_DIR}.\n\nDetails: {e}")
    st.stop()

try:
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    st.error(
        "No Groq API key found. Add GROQ_API_KEY in your Streamlit Cloud "
        "app's Settings → Secrets."
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
            response = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            answer = response.choices[0].message.content

        st.markdown(answer)

        sources = [
            (Path(doc.metadata.get("source", "unknown file")).name, doc.metadata.get("page", "?"))
            for doc in docs
        ]
        with st.expander("📄 Where this came from"):
            for source, page in sources:
                st.write(f"- **{source}**, page {page}")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
