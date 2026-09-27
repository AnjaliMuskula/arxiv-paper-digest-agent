# # graph.py
# graph.py

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from state import AgentState

from arxiv import (
    extract_arxiv_id,
    search_arxiv,
    get_paper_metadata,
    download_pdf
)

from pdf_parser import parse_pdf
from chunking import create_chunks
from embeddings import embed_texts
from vector_store import store_chunks

from prompt import build_briefing_prompt
from llm import generate_answer


# ==================================================
# 1. QUERY UNDERSTANDING
# ==================================================

def query_understanding(
    state: AgentState
):

    user_query = state["user_query"]

    arxiv_id = extract_arxiv_id(
        user_query
    )

    if arxiv_id:

        return {
            "query_type": "paper_id",
            "paper_id": arxiv_id
        }

    return {
        "query_type": "topic"
    }


# ==================================================
# 2. ARXIV RETRIEVAL
# ==================================================

def retrieve_paper(
    state: AgentState
):

    user_query = state["user_query"]

    query_type = state["query_type"]

    # Specific paper
    if query_type == "paper_id":

        paper = get_paper_metadata(
            state["paper_id"]
        )

        return {
            "selected_paper": paper
        }

    # Topic search
    papers = search_arxiv(
        user_query,
        max_results=5
    )

    if not papers:

        # Modeled as state, not an exception, so this failure path is a
        # real edge in the graph (visible in the architecture) rather than
        # something that bypasses it.
        return {
            "papers": [],
            "error": "no_candidates"
        }

    # "Many candidates" case: we trust arXiv's own relevance ranking and
    # auto-select the top result. `papers` (the full candidate list) is
    # still returned in state so the UI can show what else was found and
    # what was passed over -- see app.py.
    return {
        "papers": papers,
        "selected_paper": papers[0],
        "paper_id": papers[0]["id"]
    }


# ==================================================
# ROUTING: handle zero-candidate failure case
# ==================================================

def route_after_retrieval(state: AgentState):
    if state.get("error") == "no_candidates":
        return "no_results"
    return "continue"


def handle_no_results(state: AgentState):
    # Terminal node for the zero-candidates path -- lets the UI show a
    # clean message instead of a stack trace, and keeps the branch
    # visible in the compiled graph.
    return {
        "briefing": None,
        "error": "no_candidates",
    }


# ==================================================
# 3. FETCH PDF
# ==================================================

def fetch_pdf(
    state: AgentState
):

    paper = state["selected_paper"]

    pdf_path = download_pdf(
        paper
    )

    return {
        "pdf_path": pdf_path
    }


# ==================================================
# 4. PARSE PDF
# ==================================================

def parse_paper(
    state: AgentState
):

    text = parse_pdf(
        state["pdf_path"]
    )

    if not text.strip():

        raise ValueError(
            "Could not extract readable text."
        )

    return {
        "parsed_text": text
    }


# ==================================================
# 5. CHUNK
# ==================================================

def chunk_paper(
    state: AgentState
):

    chunks = create_chunks(
        state["parsed_text"],
        chunk_size=1200,
        overlap=200
    )

    return {
        "chunks": chunks
    }


# ==================================================
# 6. EMBED + STORE
# ==================================================

def index_paper(
    state: AgentState
):

    chunks = state["chunks"]

    embeddings = embed_texts(
        chunks
    )

    store_chunks(
        state["paper_id"],
        chunks,
        embeddings
    )

    return {}


# ==================================================
# 7. GENERATE EXECUTIVE BRIEFING
# ==================================================

# Safety cap so a very long paper doesn't blow the prompt budget or
# silently push earlier sections out of the model's effective context.
MAX_BRIEFING_CHARS = 30000


def generate_briefing(
    state: AgentState
):

    full_text = state["parsed_text"]
    was_truncated = len(full_text) > MAX_BRIEFING_CHARS
    text_for_prompt = full_text[:MAX_BRIEFING_CHARS]

    prompt = build_briefing_prompt(
        state["selected_paper"],
        text_for_prompt
    )

    briefing = generate_answer(
        prompt
    )

    if was_truncated:
        briefing += (
            "\n\n_Note: this paper is long; the briefing above is based on "
            "the first ~30,000 characters of extracted text and may miss "
            "detail from later sections._"
        )

    return {
        "briefing": briefing
    }


# ==================================================
# BUILD GRAPH
# ==================================================

def build_graph():

    graph = StateGraph(
        AgentState
    )

    # Nodes
    graph.add_node(
        "query_understanding",
        query_understanding
    )

    graph.add_node(
        "retrieve_paper",
        retrieve_paper
    )

    graph.add_node(
        "handle_no_results",
        handle_no_results
    )

    graph.add_node(
        "fetch_pdf",
        fetch_pdf
    )

    graph.add_node(
        "parse_paper",
        parse_paper
    )

    graph.add_node(
        "chunk_paper",
        chunk_paper
    )

    graph.add_node(
        "index_paper",
        index_paper
    )

    graph.add_node(
        "generate_briefing",
        generate_briefing
    )

    # Edges

    graph.add_edge(
        START,
        "query_understanding"
    )

    graph.add_edge(
        "query_understanding",
        "retrieve_paper"
    )

    graph.add_conditional_edges(
        "retrieve_paper",
        route_after_retrieval,
        {
            "continue": "fetch_pdf",
            "no_results": "handle_no_results",
        }
    )

    graph.add_edge(
        "handle_no_results",
        END
    )

    graph.add_edge(
        "fetch_pdf",
        "parse_paper"
    )

    graph.add_edge(
        "parse_paper",
        "chunk_paper"
    )

    graph.add_edge(
        "chunk_paper",
        "index_paper"
    )

    graph.add_edge(
        "index_paper",
        "generate_briefing"
    )

    graph.add_edge(
        "generate_briefing",
        END
    )

    return graph.compile()


