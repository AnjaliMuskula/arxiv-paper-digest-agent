# 1. Architecture

## High-Level Architecture

```text
                         ┌──────────────┐
                         │     USER     │
                         │  Streamlit   │
                         └──────┬───────┘
                                │
                                ▼
                    ┌─────────────────────┐
                    │ Query Understanding │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
            Topic Input                arXiv ID / URL
                 │                           │
                 ▼                           │
        ┌──────────────────┐                 │
        │  arXiv Search    │                 │
        │ Official API     │                 │
        └────────┬─────────┘                 │
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                     ┌──────────────────┐
                     │   Fetch PDF      │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Parse PDF      │
                     │  PyMuPDF / OCR   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │    Chunking      │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │   Embeddings     │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │     Chroma       │
                     │   Vector Store   │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │ Executive        │
                     │ Briefing         │
                     └────────┬─────────┘
                              │
                              ▼
                         ┌─────────┐
                         │   QA    │
                         └────┬────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │ Vector Search   │       │    BM25 Search  │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 └──────────┬──────────────┘
                            ▼
                    ┌───────────────┐
                    │  RRF Ranking  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Relevant      │
                    │ Chunks        │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │      LLM      │
                    │ Gemini Flash  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │ Grounded      │
                    │ Answer        │
                    └───────────────┘
```

| Component              | Technology                   |
| ---------------------- | ---------------------------- |
| UI                     | Streamlit                    |
| Workflow orchestration | LangGraph                    |
| Paper source           | Official arXiv API           |
| PDF parsing            | PyMuPDF                      |
| OCR fallback           | Tesseract / PyTesseract      |
| Chunking               | Custom Python chunking       |
| Embeddings             | Sentence Transformers        |
| Vector database        | Chroma                       |
| Keyword retrieval      | BM25                         |
| Hybrid ranking         | Reciprocal Rank Fusion (RRF) |
| LLM                    | Gemini                       |
| Language               | Python                       |

---

# 6. Project Structure

```text
arxiv-agent/
│
├── app.py
│
├── graph.py
├── state.py
│
├── arxiv.py
├── pdf_parser.py
├── chunking.py
├── embeddings.py
├── vector_store.py
├── hybrid_search.py
│
├── llm.py
├── prompt.py
│
├── data/
│   └── .gitkeep
│
├── chroma_db/
│   └── ...
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Module Responsibilities

| File               | Responsibility                                        |
| ------------------ | ----------------------------------------------------- |
| `app.py`           | Streamlit user interface                              |
| `graph.py`         | LangGraph workflow and nodes                          |
| `state.py`         | Shared AgentState definition                          |
| `arxiv.py`         | arXiv API search, metadata retrieval and PDF download |
| `pdf_parser.py`    | PDF text extraction and OCR fallback                  |
| `chunking.py`      | Text chunking                                         |
| `embeddings.py`    | Sentence Transformer embeddings                       |
| `vector_store.py`  | Chroma vector database                                |
| `hybrid_search.py` | BM25 + vector search + RRF                            |
| `llm.py`           | Gemini LLM integration                                |
| `prompt.py`        | Briefing and grounded QA prompts                      |
| `app.py`           | Streamlit application                                 |

---

# 3. Setup

## Install

* Python 3.10+
* Git
* Internet connection
* A Gemini API key

---

# 4. Create Virtual Environment

## Windows

```powershell
python -m venv .venv
```

### Activate

```powershell
.venv\Scripts\activate
```

## Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 5. Install Dependencies

```bash
pip install -r requirement.txt
```

---

# 6. Run the Application

## Start Streamlit

```bash
streamlit run app.py
```

The application will open in the browser.

---

# 7. Future Architecture

A future version could extend the current architecture to:

```text
                    arXiv Agent
                         │
          ┌──────────────┴──────────────┐
          │                             │
     New Paper                    Existing Paper
          │                             │
          ▼                             ▼
    Process & Index               Load Chroma
          │                             │
          └──────────────┬──────────────┘
                         ▼
                      Chat
                         │
                         ▼
                Hybrid Retrieval
                         │
                         ▼
                   Grounded LLM
                         │
                         ▼
              Answer + Citations
```
