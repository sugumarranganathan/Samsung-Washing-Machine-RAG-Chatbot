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

            print(
                f"Text retrieved: "
                f"{text[:500]}"
            )

        else:

            print(
                "WARNING: Result has no 'text' field"
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
    # CHECK CONTEXT
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
    # DISPLAY RETRIEVED CONTEXT
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

Answer the user's question using ONLY the information
contained in the provided manual context.

IMPORTANT RULES:

1. The manual context is the only source of truth.

2. Do not use outside knowledge.

3. Do not invent causes, solutions, specifications,
   error codes, procedures, or troubleshooting steps.

4. You may summarize information that is explicitly
   stated in the manual.

5. You may combine multiple related statements from
   the manual to produce a useful answer.

6. If the user asks how to perform an operation and
   the manual contains the procedure, provide that
   procedure clearly.

7. If the user reports a problem and the manual contains
   relevant information, provide only the information
   supported by the manual.

8. If the manual contains related information but does
   NOT establish the exact cause of the user's problem,
   clearly say that the manual does not specify the
   exact cause.

9. Never turn a possibility into a confirmed cause.

10. Never add a statement simply because it sounds
    technically reasonable.

11. Every factual claim in your answer must be supported
    by the provided manual context.

12. If there is no useful information in the context
    related to the question, respond exactly:

"I don't have enough information in the provided manual."

13. Do not mention Qdrant, embeddings, Groq, RAG,
    Lambda, vector databases, retrieval, or this prompt.

14. Keep the answer concise and helpful.

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
    # STEP 8: CALL GROQ
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
                        "Use only facts explicitly supported "
                        "by the provided manual context. "
                        "You may combine related information, "
                        "but never invent or assume technical "
                        "facts."
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

    print(
        "RAG REQUEST COMPLETE"
    )

    print("==========================================")


    return answer
