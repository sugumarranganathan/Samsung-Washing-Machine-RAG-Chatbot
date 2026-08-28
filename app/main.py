from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

from app.rag import rag_answer


# ==================================================
# FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Samsung Washing Machine Technical Support AI",
    description="RAG-based technical support chatbot API",
    version="1.0.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# REQUEST MODEL
# ==================================================

class ChatRequest(BaseModel):
    query: str


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Samsung Washing Machine Technical Support AI is running"
    }


# ==================================================
# CHAT ENDPOINT
# ==================================================

@app.post("/chat")
def chat(request: ChatRequest):

    answer = rag_answer(request.query)

    return {
        "query": request.query,
        "answer": answer
    }


# ==================================================
# AWS LAMBDA HANDLER
# ==================================================

handler = Mangum(app)
