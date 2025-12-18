import os
import cohere
from pydantic import BaseModel

from embeddings import get_embedding
from retriever import retrieve_context

client = cohere.Client(os.getenv("COHERE_API_KEY"))


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def ask_question(question: str) -> str:
    embedding = get_embedding(question)

    context_docs = retrieve_context(embedding, top_k=5)

    context_text = "\n\n".join(
        [doc["content"] for doc in context_docs if doc["content"]]
    )

    prompt = f"""
You are a helpful assistant for a robotics textbook website.

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
"""
    

    response = client.generate(
        model="command-light",
        prompt=prompt,
        temperature=0.3,
        max_tokens=500,
    )

    return response.generations[0].text

    


    
    


