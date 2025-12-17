from qdrant_client.http.exceptions import UnexpectedResponse
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key if qdrant_api_key else None,
)

COLLECTION_NAME = "book-docs"
VECTOR_SIZE = 1024  # Cohere embed-english-v3.0 output dimension


def ensure_collection_exists():
    """Create Qdrant collection only if it does not exist."""
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' already exists.")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )
            print(f"Collection '{COLLECTION_NAME}' created.")
        else:
            raise e



def retrieve_context(embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Retrieve top-k relevant documents from Qdrant based on embedding similarity.

    Args:
        embedding: The query embedding vector
        top_k: Number of results to return


    Returns:
        List of dicts with 'content' and 'score' keys
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
