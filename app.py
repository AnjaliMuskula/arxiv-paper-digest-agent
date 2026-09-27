# app.py

import streamlit as st

from graph import build_graph
from embeddings import embed_query
from hybrid_search import HybridRetriever
from prompt import build_qa_prompt
from llm import generate_answer


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="arXiv Paper Chat",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

defaults = {

    "candidates": [],

    "paper": None,

    "paper_text": None,

    "chunks": [],

    "briefing": None,

    "paper_id": None,

    "ready_for_qa": False,

    "conversation_history": []

}

for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# ==================================================
# HEADER
# ==================================================

st.title("📚 arXiv Paper Chat")


# ==================================================
# ADD PAPER SECTION
# ==================================================

st.subheader("➕ Add a Paper")

user_query = st.text_input(
    "Enter arXiv URL or Paper ID",
    placeholder="Example: https://arxiv.org/abs/1706.03762",
    label_visibility="collapsed"
)


if st.button(
    "➕ Add Paper",
    type="primary"
):

    if not user_query.strip():

        st.warning(
            "Please enter an arXiv URL or paper ID."
        )

        st.stop()

    try:

        # ------------------------------------------
        # Build LangGraph
        # ------------------------------------------

        graph = build_graph()

        # ------------------------------------------
        # Run LangGraph
        # ------------------------------------------

        with st.spinner(
            "Processing paper..."
        ):

            initial_state = {

                "user_query": user_query,

                "conversation_history": []

            }

            result = graph.invoke(
                initial_state
            )

        # ------------------------------------------
        # No paper found
        # ------------------------------------------

        if result.get("error") == "no_candidates":

            st.warning(
                "No matching paper found. "
                "Please check the arXiv URL or paper ID."
            )

            st.stop()

        # ------------------------------------------
        # Save paper information
        # ------------------------------------------

        st.session_state.paper = (
            result["selected_paper"]
        )

        st.session_state.paper_id = (
            result["paper_id"]
        )

        st.session_state.paper_text = (
            result["parsed_text"]
        )

        st.session_state.chunks = (
            result["chunks"]
        )

        st.session_state.candidates = (
            result.get("papers", [])
        )

        st.session_state.briefing = (
            result["briefing"]
        )

        st.session_state.ready_for_qa = True

        st.session_state.conversation_history = []

        st.success(
            "Paper added successfully!"
        )

    except Exception as e:

        st.error(
            f"Failed to process paper: {str(e)}"
        )


# ==================================================
# PAPER INFORMATION
# ==================================================

if st.session_state.paper:

    paper = st.session_state.paper

    st.divider()

    st.subheader("📄 Current Paper")

    st.write(
        f"**{paper['title']}**"
    )

    st.caption(
        f"arXiv ID: {paper['id']}"
    )


# ==================================================
# EXECUTIVE BRIEFING
# ==================================================

if st.session_state.briefing:

    with st.expander(
        "📝 View Executive Briefing",
        expanded=True
    ):

        st.markdown(
            st.session_state.briefing
        )


# ==================================================
# CHAT HISTORY
# ==================================================

for message in st.session_state.conversation_history:

    with st.chat_message("user"):

        st.write(
            message["question"]
        )

    with st.chat_message("assistant"):

        st.markdown(
            message["answer"]
        )


# ==================================================
# BOTTOM CHAT BAR
# ==================================================

if st.session_state.ready_for_qa:

    question = st.chat_input(
        "Ask something about this paper..."
    )

    if question:

        try:

            # --------------------------------------
            # USER MESSAGE
            # --------------------------------------

            with st.chat_message("user"):

                st.write(question)

            # --------------------------------------
            # HYBRID RETRIEVER
            # --------------------------------------

            retriever = HybridRetriever(
                paper_id=st.session_state.paper_id,
                chunks=st.session_state.chunks,
                embedding_model=embed_query
            )

            # --------------------------------------
            # SEARCH PAPER
            # --------------------------------------

            with st.spinner(
                "Searching the paper..."
            ):

                retrieved_chunks = (
                    retriever.search(
                        question,
                        top_k=5
                    )
                )

            if not retrieved_chunks:

                answer = (
                    "I could not find relevant information "
                    "in the paper to answer this question."
                )

            else:

                # ----------------------------------
                # GROUNDED PROMPT
                # ----------------------------------

                qa_prompt = build_qa_prompt(
                    question,
                    retrieved_chunks,
                    st.session_state.paper
                )

                # ----------------------------------
                # GENERATE ANSWER
                # ----------------------------------

                with st.spinner(
                    "Generating answer..."
                ):

                    answer = generate_answer(
                        qa_prompt
                    )

            # --------------------------------------
            # DISPLAY ANSWER
            # --------------------------------------

            with st.chat_message("assistant"):

                st.markdown(answer)

            # --------------------------------------
            # SAVE CONVERSATION
            # --------------------------------------

            st.session_state.conversation_history.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

        except Exception as e:

            st.error(
                f"Question answering failed: {str(e)}"
            )


# ==================================================
# EMPTY STATE
# ==================================================

if not st.session_state.paper:

    st.divider()

    st.info(
        "👆 Add an arXiv paper above to start chatting."
    )



