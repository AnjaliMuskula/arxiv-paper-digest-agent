from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def embed_texts(texts):

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


def embed_query(query):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    return embedding[0]