# hybrid_search.py

from rank_bm25 import BM25Okapi

from vector_store import vector_search


class HybridRetriever:

    def __init__(
        self,
        paper_id: str,
        chunks: list,
        embedding_model
    ):

        self.paper_id = paper_id
        self.chunks = chunks
        self.embedding_model = embedding_model

        # Prepare BM25
        tokenized_chunks = [
            chunk.lower().split()
            for chunk in chunks
        ]

        self.bm25 = BM25Okapi(
            tokenized_chunks
        )

    def bm25_search(
        self,
        query: str,
        top_k: int = 5
    ):

        scores = self.bm25.get_scores(
            query.lower().split()
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked_indices[:top_k]

    def vector_search_indices(
        self,
        query: str,
        top_k: int = 5
    ):

        query_embedding = self.embedding_model(
            query
        )

        results = vector_search(
            self.paper_id,
            query_embedding,
            top_k
        )

        vector_indices = []

        for chunk_id in results["ids"]:

            # ID format:
            # paper_id_chunk_5

            index = int(
                chunk_id.split("_chunk_")[-1]
            )

            vector_indices.append(index)

        return vector_indices

    def reciprocal_rank_fusion(
        self,
        vector_indices,
        bm25_indices,
        k=60
    ):

        scores = {}

        # Vector ranking
        for rank, index in enumerate(
            vector_indices
        ):

            scores[index] = scores.get(
                index,
                0
            ) + 1 / (k + rank + 1)

        # BM25 ranking
        for rank, index in enumerate(
            bm25_indices
        ):

            scores[index] = scores.get(
                index,
                0
            ) + 1 / (k + rank + 1)

        ranked_indices = sorted(
            scores,
            key=scores.get,
            reverse=True
        )

        return ranked_indices

    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        # ------------------------------------
        # 1. Vector search
        # ------------------------------------

        vector_indices = (
            self.vector_search_indices(
                query,
                top_k
            )
        )

        # ------------------------------------
        # 2. BM25 search
        # ------------------------------------

        bm25_indices = self.bm25_search(
            query,
            top_k
        )

        # ------------------------------------
        # 3. RRF
        # ------------------------------------

        ranked_indices = (
            self.reciprocal_rank_fusion(
                vector_indices,
                bm25_indices
            )
        )

        # ------------------------------------
        # 4. Get final chunks
        # ------------------------------------

        retrieved_chunks = [
            self.chunks[i]
            for i in ranked_indices[:top_k]
        ]

        return retrieved_chunks