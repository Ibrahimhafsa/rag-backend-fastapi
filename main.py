from retriever import ensure_collection_exists
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from api import AskRequest, AskResponse, ask_question

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="RAG Chatbot API",
    description="FastAPI backend for RAG chatbot integrated with Cohere and Qdrant",
    version="1.0.0",
)
@app.on_event("startup")
async def startup_event():
    ensure_collection_exists()


# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint for deployment monitoring."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """
    Process a question through the RAG pipeline.

    Request:
        {
            "question": "What is a humanoid robot?"
        }

    Response:
        {
            "answer": "A humanoid robot is a robot designed to resemble a human..."
        }

    Raises:
        HTTPException: If question is empty or processing fails
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        answer = ask_question(request.question)
        return AskResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
