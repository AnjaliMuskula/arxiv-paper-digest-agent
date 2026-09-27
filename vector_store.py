# vector_store.py

import chromadb


CHROMA_PATH = "./chroma_db"

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)


def get_collection(paper_id: str):

    safe_id = paper_id.replace(".", "_").replace("/", "_")

    collection_name = f"paper_{safe_id}"

    return client.get_or_create_collection(
        name=collection_name
    )


def store_chunks(
    paper_id: str,
    chunks: list,
    embeddings
):

    collection = get_collection(paper_id)

    # Clear previous version if the same paper
    # is processed again.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(
            ids=existing["ids"]
        )

    ids = [
        f"{paper_id}_chunk_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    return collection


def vector_search(
    paper_id: str,
    query_embedding,
    top_k: int = 5
):

    collection = get_collection(paper_id)

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return {
        "ids": results["ids"][0],
        "documents": results["documents"][0]
    }