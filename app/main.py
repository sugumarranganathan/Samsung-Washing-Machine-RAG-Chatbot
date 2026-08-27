
from fastapi import FastAPI
from pydantic import BaseModel
from mangum import Mangum

from app.rag import rag_answer


app = FastAPI(
    title="Samsung Washing Machine Technical Support AI",
    description="RAG-based technical support chatbot API",
    version="1.0.0"
)


class ChatRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {
        "message": "Samsung Washing Machine Technical Support AI is running"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    answer = rag_answer(request.query)

    return {
        "query": request.query,
        "answer": answer
    }


# AWS Lambda handler
handler = Mangum(app)
