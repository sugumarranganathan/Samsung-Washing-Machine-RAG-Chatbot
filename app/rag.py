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

MODEL_PATH = "/opt/models/all-MiniLM-L6-v2"

print("🔄 Loading embedding model...")

embedding_model = SentenceTransformer(
    MODEL_PATH
)

print("✅ Embedding model loaded successfully")


# --------------------------------------------------
# Qdrant client
# --------------------------------------------------

print("🔄 Creating Qdrant client...")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60
)

print("✅ Qdrant client created")


# --------------------------------------------------
# Groq client
# --------------------------------------------------

print("🔄 Creating Groq client...")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print("✅ Groq client created")


# --------------------------------------------------
# Qdrant collection
# --------------------------------------------------

COLLECTION_NAME = "samsung_washing_machine"


# --------------------------------------------------
# RAG function
# --------------------------------------------------

def rag_answer(question: str, top_k: int = 3):

    print("========================================")
    print("🚀 RAG request started")
    print("Question:", question)
    print("========================================")


    # --------------------------------------------------
    # Create query embedding
    # --------------------------------------------------

    print("🔄 Creating query embedding...")

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    print("✅ Query embedding created")
    print("Embedding dimension:", len(query_embedding))


    # --------------------------------------------------
    # Search Qdrant
    # --------------------------------------------------

    print("🔎 Searching Qdrant...")
    print("Collection:", COLLECTION_NAME)
    print("Top K:", top_k)

    search_results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        with_payload=True
    )

    print("✅ Qdrant search completed")

    print(
        "Number of results:",
        len(search_results.points)
    )


    # --------------------------------------------------
    # Display retrieved results
    # --------------------------------------------------

    context_parts = []

    for i, result in enumerate(search_results.points):

        print("----------------------------------------")
        print(f"RESULT {i + 1}")

        print("Score:", result.score)

        payload = result.payload or {}

        print("Payload:", payload)

        text = payload.get("text", "")

        if text:

            context_parts.append(text)

            print(
                "Text retrieved:",
                text[:500]
            )

        else:

            print("⚠️ No 'text' field found in payload")


    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = "\n\n".join(context_parts)

    print("----------------------------------------")
    print("📚 Context length:", len(context))
    print("📚 Number of context documents:", len(context_parts))


    if not context:

        print("⚠️ No context retrieved from Qdrant")

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # --------------------------------------------------
    # RAG prompt
    # --------------------------------------------------

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


    # --------------------------------------------------
    # Generate answer using Groq
    # --------------------------------------------------

    print("🤖 Sending request to Groq...")

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

    print("✅ Groq response received")


    # --------------------------------------------------
    # Extract answer
    # --------------------------------------------------

    answer = response.choices[0].message.content

    print("📝 Final answer:")
    print(answer)

    print("========================================")
    print("✅ RAG request completed")
    print("========================================")


    return answer
