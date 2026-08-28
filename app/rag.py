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

TOP_K = 3


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

print("STEP 1: Loading Sentence Transformer model")

start_time = time.time()

try:

    embedding_model = SentenceTransformer(
        MODEL_PATH
    )

except Exception as e:

    print(
        f"ERROR loading embedding model: {repr(e)}"
    )

    raise


print(
    f"STEP 1 COMPLETE: Model loaded in "
    f"{time.time() - start_time:.2f} seconds"
)


# ==================================================
# QDRANT CLIENT
# ==================================================

print("STEP 2: Creating Qdrant client")

try:

    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=15
    )

except Exception as e:

    print(
        f"ERROR creating Qdrant client: {repr(e)}"
    )

    raise


print(
    "STEP 2 COMPLETE: Qdrant client created"
)


# ==================================================
# GROQ CLIENT
# ==================================================

print("STEP 3: Creating Groq client")

try:

    groq_client = Groq(
        api_key=GROQ_API_KEY,
        timeout=30.0,
        max_retries=0
    )

except Exception as e:

    print(
        f"ERROR creating Groq client: {repr(e)}"
    )

    raise


print(
    "STEP 3 COMPLETE: Groq client created"
)


# ==================================================
# RAG FUNCTION
# ==================================================

def rag_answer(
    question: str,
    top_k: int = TOP_K
):

    print("==========================================")
    print("RAG REQUEST STARTED")
    print(f"Question: {question}")
    print(f"Top K: {top_k}")
    print("==========================================")


    # ==================================================
    # STEP 4: CREATE QUERY EMBEDDING
    # ==================================================

    print("STEP 4: Creating query embedding")

    start_time = time.time()

    try:

        query_embedding = embedding_model.encode(
            question
        ).tolist()

    except Exception as e:

        print(
            f"ERROR in embedding: {repr(e)}"
        )

        return (
            "Error creating the query embedding."
        )


    print(
        f"STEP 4 COMPLETE: Embedding created in "
        f"{time.time() - start_time:.2f} seconds"
    )

    print(
        f"Embedding dimensions: "
        f"{len(query_embedding)}"
    )


    # ==================================================
    # STEP 5: SEARCH QDRANT
    # ==================================================

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

        print(
            f"ERROR in Qdrant: {repr(e)}"
        )

        return (
            "Unable to search the knowledge base."
        )


    print(
        f"STEP 5 COMPLETE: Qdrant search completed in "
        f"{time.time() - start_time:.2f} seconds"
    )

    print(
        f"Number of results: "
        f"{len(search_results.points)}"
    )


    # ==================================================
    # STEP 6: BUILD CONTEXT
    # ==================================================

    print("STEP 6: Building context")

    context_parts = []


    for index, result in enumerate(
        search_results.points,
        start=1
    ):

        payload = result.payload or {}

        text = payload.get(
            "text",
            ""
        )

        score = getattr(
            result,
            "score",
            None
        )


        print(
            f"RESULT {index}"
        )

        print(
            f"Score: {score}"
        )

        print(
            f"Text length: {len(text)}"
        )


        if text:

            context_parts.append(text)

            # Print retrieved text for debugging

            print(
                f"Text retrieved: "
                f"{text[:500]}"
            )


        print("------------------------------------------")


    context = "\n\n".join(
        context_parts
    )


    print(
        f"STEP 6 COMPLETE: Context length = "
        f"{len(context)} characters"
    )

    print(
        f"Number of context documents: "
        f"{len(context_parts)}"
    )


    # ==================================================
    # STEP 6.1: CHECK CONTEXT
    # ==================================================

    if not context.strip():

        print(
            "WARNING: Qdrant returned no text context"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ==================================================
    # STEP 6.2: PRINT CONTEXT
    # ==================================================

    print("========== RETRIEVED CONTEXT ==========")

    print(context)

    print("========================================")


    # ==================================================
    # STEP 7: RAG PROMPT
    # ==================================================

    print("STEP 7: Preparing RAG prompt")


    prompt = f"""
You are a Samsung Washing Machine Technical Support Assistant.

Your task is to answer the user's question using ONLY
the information contained in the provided manual context.

IMPORTANT INSTRUCTIONS:

1. Use only information found in the manual context.

2. You are allowed to combine multiple related pieces
   of information from the context to form a useful answer.

3. If the user describes a problem, identify the
   relevant operating instructions, troubleshooting
   instructions, error-code information, maintenance
   information, or safety information in the context.

4. Do NOT invent technical information.

5. Do NOT assume information that is not present in
   the manual.

6. Do NOT provide Samsung specifications, procedures,
   error codes, or troubleshooting steps that are not
   explicitly supported by the context.

7. If the context contains relevant information,
   provide a concise and practical answer based on it.

8. If the context does not contain enough relevant
   information to answer the question, respond exactly:

"I don't have enough information in the provided manual."

9. Do not mention the RAG system, Qdrant, embeddings,
   vector database, Groq, or this prompt.

10. Treat the manual context as the only source of truth.

MANUAL CONTEXT:
----------------
{context}
----------------

USER QUESTION:
{question}

ANSWER:
"""


    print(
        f"STEP 7 COMPLETE: Prompt prepared "
        f"({len(prompt)} characters)"
    )


    # ==================================================
    # STEP 8: SEND REQUEST TO GROQ
    # ==================================================

    print("STEP 8: Sending request to Groq")

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
                        "Answer using only the provided manual "
                        "context. You may combine related "
                        "information from the context, but "
                        "never invent technical information."
                    )
                },

                {
                    "role": "user",

                    "content": prompt
                }

            ],

            temperature=0.1
        )

    except Exception as e:

        print(
            f"ERROR in Groq: {repr(e)}"
        )

        return (
            "Unable to generate an answer "
            "from the AI model."
        )


    print(
        f"STEP 8 COMPLETE: Groq response received in "
        f"{time.time() - start_time:.2f} seconds"
    )


    # ==================================================
    # STEP 9: EXTRACT ANSWER
    # ==================================================

    print("STEP 9: Extracting final answer")

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as e:

        print(
            f"ERROR extracting Groq response: "
            f"{repr(e)}"
        )

        return (
            "Unable to extract the AI response."
        )


    # ==================================================
    # STEP 10: FINAL ANSWER
    # ==================================================

    print("==========================================")
    print("FINAL ANSWER:")
    print(answer)
    print("==========================================")

    print("RAG REQUEST COMPLETE")

    print("==========================================")


    return answer
