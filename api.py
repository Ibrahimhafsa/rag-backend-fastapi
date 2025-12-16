import os
import cohere
from pydantic import BaseModel
from embeddings import get_embedding
from retriever import retrieve_context

client = cohere.Client(os.getenv("COHERE_API_KEY"))


class AskRequest(BaseModel):
    """Request schema for /ask endpoint."""
    question: str


class AskResponse(BaseModel):
    """Response schema for /ask endpoint."""
    answer: str


def ask_question(question: str) -> str:
    """
    Process a question using RAG pipeline:
    1. Embed the question with Cohere
    2. Retrieve relevant context from Qdrant
    3. Generate answer using Cohere with retrieved context

    Args:
        question: The user's question

    Returns:
        Generated answer string
    """
    # Step 1: Embed the question
    embedding = get_embedding(question)

    # Step 2: Retrieve relevant context
    context_docs = retrieve_context(embedding, top_k=5)

    # Step 3: Format context for the prompt
    context_text = "\n\n".join(
        [f"Source: {doc['source']}\n{doc['content']}" for doc in context_docs]
    )

    # Step 4: Generate answer using Cohere
    prompt = f"""You are a helpful assistant for a robotics book documentation site.
Answer the user's question based on the provided context from the book.
If the context doesn't contain relevant information, say so honestly.

Context from the book:
{context_text}

User question: {question}

Answer:"""

    response = client.chat(
        model="command-r-v1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.message.content[0].text
