# A/L Business Studies AI Tutor (Local RAG Chatbot)

A free, offline AI study assistant for the Sri Lankan Advanced Level
Business Studies syllabus. Ask it any syllabus question and it answers
using **your actual syllabus documents** — not guesses — via a
technique called **RAG (Retrieval-Augmented Generation)**.

Everything runs on your own computer. No API keys, no subscriptions,
no internet required once set up.

---

## How it works (in plain English)

1. Your syllabus PDFs are cut into small chunks and turned into
   "vectors" (number representations of meaning) — this is the
   `ingest.py` step, done once.
2. When you ask a question, the app finds the 3-4 chunks most related
   to your question.
3. Those chunks + your question are sent to a small AI model running
   locally (via **Ollama**), which writes an answer using only that
   material.
4. The app shows the answer plus which document/page it came from.

---

## 1. Install the tools

**Python** — install Python 3.10 or newer from https://python.org
(tick "Add Python to PATH" during install on Windows).

**Ollama** — install from https://ollama.com (free, one-click
installer for Windows/Mac/Linux). This is what runs the AI model
locally.

After installing Ollama, open a terminal (Command Prompt / PowerShell
on Windows, Terminal on Mac) and download the two models you need:

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

`llama3.2:3b` is small enough to run reasonably on a normal laptop
(no GPU needed). If your PC is low on RAM (under 8GB), you can try an
even lighter model like `qwen2.5:1.5b` instead — just change
`LLM_MODEL` in `app.py` to match.

---

## 2. Install the Python packages

Open a terminal inside this project folder and run:

```bash
pip install -r requirements.txt
```

(If you want to keep things tidy, you can first create a virtual
environment with `python -m venv venv` and activate it — optional but
good practice to mention in your CV/README too.)

---

## 3. Add your syllabus content

Put PDFs into the `data/` folder. Good sources for genuine, free,
official Sri Lankan A/L Business Studies material:

- The **Department of Examinations, Sri Lanka** website — publishes
  official past papers and marking schemes.
- The **National Institute of Education (NIE)** website — publishes
  the official syllabus and teacher's/resource guides.
- Your own class notes or textbook chapters you're allowed to use.

Search for "Sri Lanka A/L Business Studies syllabus NIE pdf" and
"Department of Examinations Sri Lanka past papers Business Studies"
to find current links, since these pages get restructured
periodically.

The more complete and well-organized your PDFs are, the better your
assistant's answers will be — this step matters more than any code.

---

## 4. Build the knowledge base

```bash
python ingest.py
```

This reads every PDF in `data/`, chunks it, embeds it, and saves it
into a local folder called `chroma_db/`. Re-run this any time you add
new PDFs.

---

## 5. Run the assistant

Make sure Ollama is running in the background (it usually starts
automatically after install), then:

```bash
streamlit run app.py
```

This opens a chat page in your browser at `http://localhost:8501`.
Type a question like *"Explain the difference between a sole
proprietorship and a partnership"* and it will answer using your
syllabus material, with sources shown below the answer.

---

## 6. Improving accuracy

- If answers feel too vague, lower `chunk_size` in `ingest.py` (e.g.
  to 500) so chunks are more focused, then re-run `ingest.py`.
- If answers miss context, raise `k` in `app.py`'s
  `search_kwargs={"k": 4}` to retrieve more chunks (try 6).
- Test it against real past-paper questions and compare to the
  official marking scheme — this is a great thing to describe in your
  CV/portfolio ("evaluated against past-paper marking schemes").

---

## 7. Turning this into a CV project

- Push this folder to a **GitHub repository** with a clear README
  (you can adapt this one).
- Record a **1-2 minute screen recording** showing you asking it a
  few real syllabus questions — link it from your CV/portfolio since
  a local app can't easily be hosted live for free.
- On your CV, describe it concretely, e.g.:
  *"Built a local RAG-based study assistant for A/L Business Studies
  using LangChain, ChromaDB, and a locally-hosted LLM (Ollama/Llama
  3.2), retrieving syllabus content to ground answers and reduce
  hallucination."*
- Good to mention in interviews: why RAG instead of just prompting an
  LLM directly (accuracy, staying within syllabus scope, avoiding
  hallucinated facts, citing sources).

---

## Project structure

```
al-business-studies-ai-tutor/
├── data/            <- put your syllabus/past-paper PDFs here
├── chroma_db/        <- auto-created vector database (after ingest.py)
├── ingest.py         <- builds the knowledge base from your PDFs
├── app.py            <- the Streamlit chat app
├── requirements.txt
└── README.md
```

## Optional next steps (great for making the project stand out)

- Extend it to Accounting or Economics too — just swap the PDFs and
  rebuild the database (or run multiple databases and let the user
  pick a subject).
- Add a "past paper mode" that quizzes the student instead of just
  answering.
- Deploy the UI (without the local LLM) to Streamlit Community Cloud
  as a portfolio demo, using a free cloud LLM API as a fallback for
  online visitors — while keeping the local version as your main,
  fully free build.
