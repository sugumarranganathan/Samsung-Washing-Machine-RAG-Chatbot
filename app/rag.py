import os

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq


# --------------------------------------------------
# Environment variables
# --------------------------------------------------

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# --------------------------------------------------
# Local Sentence Transformer model
# --------------------------------------------------
# The model is downloaded during Docker build
# and stored inside the Docker image.
# Lambda loads it locally without contacting
# Hugging Face during execution.

MODEL_PATH = "/opt/models/all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(
    MODEL_PATH
)


# --------------------------------------------------
# Qdrant client
# --------------------------------------------------

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)


# --------------------------------------------------
# Groq client
# --------------------------------------------------

groq_client = Groq(
    api_key=GROQ_API_KEY
)


# --------------------------------------------------
# Qdrant collection
# --------------------------------------------------

COLLECTION_NAME = "samsung_washing_machine"


# --------------------------------------------------
# RAG function
# --------------------------------------------------

def rag_answer(question: str, top_k: int = 3):

    # ----------------------------------------------
    # Create query embedding
    # ----------------------------------------------

    query_embedding = embedding_model.encode(
        question
    ).tolist()


    # ----------------------------------------------
    # Search Qdrant
    # ----------------------------------------------

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )


    # ----------------------------------------------
    # Build context from retrieved documents
    # ----------------------------------------------

    context_parts = []

    for result in search_results.points:

        payload = result.payload or {}

        text = payload.get("text", "")

        if text:
            context_parts.append(text)


    context = "\n\n".join(context_parts)


    # ----------------------------------------------
    # RAG prompt
    # ----------------------------------------------

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


    # ----------------------------------------------
    # Generate answer using Groq
    # ----------------------------------------------

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful Samsung washing machine "
                    "technical support assistant. "
                    "Use only the provided manual context."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )


    # ----------------------------------------------
    # Return answer
    # ----------------------------------------------

    return response.choices[0].message.content
