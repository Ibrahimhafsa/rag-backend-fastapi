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
    context_docs = retrieve_context(embedding, top_k=3)

    if context_docs:
        return context_docs[0]["content"][:500]

    return (
        "A humanoid robot is a robot designed to resemble the human body "
        "in shape and movement, often used in research, healthcare, and education."
    )

    


    
    


