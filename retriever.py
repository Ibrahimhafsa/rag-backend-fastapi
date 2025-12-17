import os
from qdrant_client import QdrantClient

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None,
)

COLLECTION_NAME = "book-docs"


def retrieve_context(embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k relevant documents from Qdrant.
    Assumes collection already exists.
    """

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
    )

    context = []
    for hit in results:
        context.append({
            "content": hit.payload.get("content", ""),
            "score": hit.score,
            "source": hit.payload.get("source", "unknown"),
        })

    return context
