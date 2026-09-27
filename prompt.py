# prompts.py


def build_briefing_prompt(paper: dict, paper_text: str) -> str:
    """
    Create the prompt used to generate the executive briefing.
    """

    return f"""
You are an AI research paper analyst.

You are given the metadata and extracted text of an arXiv research paper.

Your task is to create a structured executive briefing.

IMPORTANT:
- Use only the information provided below.
- Do not invent facts.
- Do not add information from outside knowledge.
- If some information is unavailable, explicitly say "Not provided in the paper."
- Keep the explanation understandable to a technical reader.

PAPER METADATA
--------------

Title:
{paper.get("title", "Not provided")}

Authors:
{", ".join(paper.get("authors", []))}

arXiv ID:
{paper.get("id", "Not provided")}

Published:
{paper.get("published", "Not provided")}

Link:
{paper.get("abs_url", "Not provided")}


PAPER TEXT
----------

{paper_text}


GENERATE THE EXECUTIVE BRIEFING USING THIS STRUCTURE:

# Executive Briefing

## 1. Paper Information
- Title:
- Authors:
- arXiv ID:
- Published:
- Link:

## 2. Why This Paper Matters
Explain in plain English what problem this paper addresses
and why the work is useful or important.

## 3. Problem Statement
Clearly explain the problem the researchers are trying to solve.

## 4. Method / Approach
Explain the proposed approach using concise bullet points.

## 5. Key Results / Claims
List the important experimental results, findings,
or claims reported by the authors.

## 6. Limitations
Explicitly describe limitations mentioned by the authors.
Do not invent limitations.

## 7. Suggested Follow-up Questions
Provide 3-5 useful questions that someone could ask
about this paper.

Remember:
Only use information supported by the paper.
"""


def build_qa_prompt(
    question: str,
    retrieved_chunks: list,
    paper: dict
) -> str:
    """
    Create a grounded RAG prompt for paper QA.
    """

    context = "\n\n".join(
        [
            f"--- Paper Excerpt {i + 1} ---\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ]
    )

    return f"""
You are a research paper question-answering assistant.

You must answer the user's question using ONLY
the retrieved excerpts from the paper.

IMPORTANT GROUNDING RULES:

1. Use only the provided paper excerpts.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer cannot be found in the provided excerpts,
   say:

   "The paper does not provide enough information
   in the retrieved sections to answer this question."

5. If the retrieved excerpts partially answer the question,
   clearly explain what is supported and what is not.
6. Keep the answer concise but technically accurate.
7. When possible, mention which paper excerpt supports
   the answer.

PAPER
-----

Title:
{paper.get("title", "Unknown")}

arXiv ID:
{paper.get("id", "Unknown")}


RETRIEVED PAPER EXCERPTS
------------------------

{context}


USER QUESTION
-------------

{question}


ANSWER
------

Provide a grounded answer based only on the retrieved
paper excerpts.
"""