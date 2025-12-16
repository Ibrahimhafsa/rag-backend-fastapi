import os
import cohere
from dotenv import load_dotenv
load_dotenv()
client = cohere.Client(os.getenv("COHERE_API_KEY"))


def get_embedding(text: str) -> list[float]:
    """
    Get embedding for a single text using Cohere.

    Args:
        text: The text to embed

    Returns:
        List of floats representing the embedding vector
    """
    response = client.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_query"
    )
    return response.embeddings[0]



def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for multiple texts using Cohere (for bulk indexing).

    Args:
        texts: List of texts to embed

    Returns:
        List of embedding vectors
    """
    response = client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document"
    )
    return response.embeddings
