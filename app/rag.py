
import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq


# -----------------------------
# Environment variables
# -----------------------------

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# -----------------------------
# Clients
# -----------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

groq_client = Groq(
    api_key=GROQ_API_KEY
)


COLLECTION_NAME = "samsung_washing_machine"


# -----------------------------
# RAG function
# -----------------------------

def rag_answer(question, top_k=3):

    # Create query embedding
    query_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search Qdrant
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )

    # Build context
    context_parts = []

    for result in search_results.points:
        text = result.payload.get("text", "")
        context_parts.append(text)

    context = "\n\n".join(context_parts)

    # RAG prompt
    prompt = f"""
You are a Samsung Washing Machine Technical Support Assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:
"I don't have enough information in the provided manual."

Do not invent technical information.

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

    # Generate answer using Groq
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful technical support assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
