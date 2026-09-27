# state.py

from typing import TypedDict, List, Dict


class AgentState(TypedDict, total=False):

    # -------------------------
    # User
    # -------------------------

    user_query: str
    query_type: str

    # -------------------------
    # arXiv
    # -------------------------

    papers: List[Dict]
    selected_paper: Dict
    paper_id: str

    # -------------------------
    # PDF
    # -------------------------

    pdf_path: str
    parsed_text: str

    # -------------------------
    # RAG
    # -------------------------

    chunks: List[str]
    retrieved_chunks: List[str]

    # -------------------------
    # Summary
    # -------------------------

    briefing: str

    # -------------------------
    # QA
    # -------------------------

    current_question: str
    conversation_history: List[Dict]
    answer: str

    # -------------------------
    # Error
    # -------------------------

    error: str

# AgentState
# │
# ├── user_query
# ├── selected_paper
# ├── pdf_path
# ├── parsed_text
# ├── chunks
# ├── retrieved_chunks
# ├── briefing
# ├── current_question
# └── answer