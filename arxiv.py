# 1. Detect paper ID
# 2. Search by topic
# 3. Get metadata
# 4. Fetch PDF

import os
import re
import requests
import feedparser
from urllib.parse import quote


ARXIV_API_URL = "https://export.arxiv.org/api/query"
DOWNLOAD_DIR = "data"


def extract_arxiv_id(user_input: str):
    """
    Extract arXiv ID from:
    - 1706.03762
    - arXiv:1706.03762
    - https://arxiv.org/abs/1706.03762
    """

    pattern = r"(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?"

    match = re.search(pattern, user_input)

    if match:
        return match.group(1)

    return None


def search_arxiv(topic: str, max_results: int = 5):
    """
    Search arXiv using the official API.
    """

    params = {
        "search_query": f"all:{quote(topic)}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []

    for entry in feed.entries:

        arxiv_id = entry.id.split("/abs/")[-1]

        paper = {
            "id": arxiv_id,
            "title": entry.title.strip(),
            "authors": [
                author.name
                for author in entry.authors
            ],
            "summary": entry.summary.strip(),
            "published": entry.published,
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
        }

        papers.append(paper)

    return papers


def get_paper_metadata(arxiv_id: str):
    """
    Fetch metadata for a specific arXiv paper.
    """

    params = {
        "id_list": arxiv_id
    }

    response = requests.get(
        ARXIV_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    feed = feedparser.parse(response.text)

    if not feed.entries:
        raise ValueError(
            f"No paper found for arXiv ID: {arxiv_id}"
        )

    entry = feed.entries[0]

    return {
        "id": arxiv_id,
        "title": entry.title.strip(),
        "authors": [
            author.name
            for author in entry.authors
        ],
        "summary": entry.summary.strip(),
        "published": entry.published,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        "abs_url": f"https://arxiv.org/abs/{arxiv_id}",
    }


def download_pdf(paper: dict):
    """
    Download the selected paper PDF.
    """

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    arxiv_id = paper["id"]

    safe_id = arxiv_id.replace("/", "_")

    pdf_path = os.path.join(
        DOWNLOAD_DIR,
        f"{safe_id}.pdf"
    )

    response = requests.get(
        paper["pdf_url"],
        timeout=60
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "content-type",
        ""
    ).lower()

    if "pdf" not in content_type:
        raise ValueError(
            "Downloaded content is not a PDF."
        )

    with open(pdf_path, "wb") as file:
        file.write(response.content)

    return pdf_path