import os
import time

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq


# ==================================================
# ENVIRONMENT VARIABLES
# ==================================================

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_PATH = "/opt/models/all-MiniLM-L6-v2"

COLLECTION_NAME = "samsung_washing_machine"


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

print("STEP 1: Loading Sentence Transformer model")

start_time = time.time()

embedding_model = SentenceTransformer(
    MODEL_PATH
)

print(
    f"STEP 1 COMPLETE: Model loaded in "
    f"{time.time() - start_time:.2f} seconds"
)


# ==================================================
# QDRANT CLIENT
# ==================================================

print("STEP 2: Creating Qdrant client")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=15
)

print("STEP 2 COMPLETE: Qdrant client created")


# ==================================================
# GROQ CLIENT
# ==================================================

print("STEP 3: Creating Groq client")

groq_client = Groq(
    api_key=GROQ_API_KEY,
    timeout=30.0,
    max_retries=0
)

print("STEP 3 COMPLETE: Groq client created")


# ==================================================
# RAG FUNCTION
# ==================================================

def rag_answer(question: str, top_k: int = 3):

    print("==========================================")
    print("RAG REQUEST STARTED")
    print(f"Question: {question}")
    print("==========================================")


    # ----------------------------------------------
    # CREATE EMBEDDING
    # ----------------------------------------------

    print("STEP 4: Creating query embedding")

    start_time = time.time()

    try:

        query_embedding = embedding_model.encode(
            question
        ).tolist()

    except Exception as e:

        print(f"ERROR in embedding: {repr(e)}")

        return "Error creating query embedding."

    print(
        f"STEP 4 COMPLETE: Embedding created in "
        f"{time.time() - start_time:.2f} seconds"
    )

    print(f"Embedding dimensions: {len(query_embedding)}")


    # ----------------------------------------------
    # QDRANT SEARCH
    # ----------------------------------------------

    print("STEP 5: Searching Qdrant")

    start_time = time.time()

    try:

        search_results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )

    except Exception as e:

        print(f"ERROR in Qdrant: {repr(e)}")

        return "Unable to search the knowledge base."


    print(
        f"STEP 5 COMPLETE: Qdrant search completed in "
        f"{time.time() - start_time:.2f} seconds"
    )

    print(
        f"Number of results: "
        f"{len(search_results.points)}"
    )


    # ----------------------------------------------
    # BUILD CONTEXT
    # ----------------------------------------------

    print("STEP 6: Building context")

    context_parts = []


    for index, result in enumerate(
        search_results.points,
        start=1
    ):

        payload = result.payload or {}

        text = payload.get("text", "")

        print(
            f"Result {index}: "
            f"score={result.score}, "
            f"text_length={len(text)}"
        )

        if text:

            context_parts.append(text)


    context = "\n\n".join(context_parts)


    print(
        f"STEP 6 COMPLETE: Context length = "
        f"{len(context)} characters"
    )


    # ----------------------------------------------
    # CHECK CONTEXT
    # ----------------------------------------------

    if not context:

        print("WARNING: Qdrant returned no text context")

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ----------------------------------------------
    # RAG PROMPT
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
    # GROQ REQUEST
    # ----------------------------------------------

    print("STEP 7: Sending request to Groq")

    start_time = time.time()

    try:

        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful Samsung washing "
                        "machine technical support assistant. "
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

    except Exception as e:

        print(f"ERROR in Groq: {repr(e)}")

        return "Unable to generate an answer from the AI model."


    print(
        f"STEP 7 COMPLETE: Groq response received in "
        f"{time.time() - start_time:.2f} seconds"
    )


    # ----------------------------------------------
    # FINAL ANSWER
    # ----------------------------------------------

    answer = response.choices[0].message.content

    print("STEP 8: Answer generated")

    print(f"Answer: {answer}")

    print("==========================================")
    print("RAG REQUEST COMPLETE")
    print("==========================================")


    return answer
