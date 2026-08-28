import os
import time
import re

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

# Number of chunks retrieved from Qdrant
TOP_K = 3

# Minimum similarity score required for the
# best retrieved document to be considered relevant.
#
# This helps prevent unrelated questions from being
# answered using weakly related washing-machine chunks.
SCORE_THRESHOLD = 0.45

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"


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
# TEXT CLEANING
# ==================================================

def clean_text(text: str) -> str:
    """
    Remove unwanted control characters from retrieved
    document text while preserving normal whitespace.
    """

    if not text:
        return ""

    # Remove ASCII control characters except:
    # newline, carriage return and tab.
    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    return text.strip()


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
    print(f"Score threshold: {SCORE_THRESHOLD}")
    print("==========================================")


    # ==================================================
    # VALIDATE QUESTION
    # ==================================================

    if not isinstance(question, str):

        print(
            "WARNING: Question is not a string"
        )

        return (
            "Please provide a valid question."
        )


    question = question.strip()


    if not question:

        print(
            "WARNING: Empty question received"
        )

        return (
            "Please enter a question about the "
            "washing machine."
        )


    # Prevent unnecessarily large requests
    if len(question) > 1000:

        print(
            "WARNING: Question exceeded 1000 characters"
        )

        return (
            "Please keep your question under "
            "1000 characters."
        )


    # ==================================================
    # STEP 4: CREATE QUERY EMBEDDING
    # ==================================================

    print("STEP 4: Creating query embedding")

    start_time = time.time()

    try:

        query_embedding = embedding_model.encode(
            question,
            normalize_embeddings=True
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


    points = search_results.points

    print(
        f"Number of results: {len(points)}"
    )


    # ==================================================
    # STEP 5A: CHECK RETRIEVAL RELEVANCE
    # ==================================================

    if not points:

        print(
            "WARNING: Qdrant returned no results"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    best_score = getattr(
        points[0],
        "score",
        0.0
    )

    print(
        f"Best similarity score: {best_score}"
    )


    if best_score is None:

        best_score = 0.0


    if best_score < SCORE_THRESHOLD:

        print(
            "WARNING: Best result is below "
            "the relevance threshold."
        )

        print(
            f"Best score: {best_score}"
        )

        print(
            f"Required score: {SCORE_THRESHOLD}"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ==================================================
    # STEP 6: BUILD CONTEXT
    # ==================================================

    print("STEP 6: Building context")

    context_parts = []

    source_names = []


    for index, result in enumerate(
        points,
        start=1
    ):

        payload = result.payload or {}

        raw_text = payload.get(
            "text",
            ""
        )

        text = clean_text(
            raw_text
        )


        score = getattr(
            result,
            "score",
            None
        )


        source = payload.get(
            "source",
            "Unknown document"
        )


        print(
            f"RESULT {index}"
        )

        print(
            f"Score: {score}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Text length: {len(text)}"
        )


        if text:

            context_parts.append(
                f"[Document Chunk {index}]\n{text}"
            )

            if source not in source_names:

                source_names.append(
                    source
                )

            print(
                f"Text retrieved: "
                f"{text[:500]}"
            )

        else:

            print(
                "WARNING: Result has no usable text"
            )


        print(
            "------------------------------------------"
        )


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
            "WARNING: Qdrant returned no usable "
            "text context"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ==================================================
    # DISPLAY RETRIEVED CONTEXT
    # ==================================================

    print(
        "========== RETRIEVED CONTEXT =========="
    )

    print(context)

    print(
        "========================================"
    )


    # ==================================================
    # STEP 7: STRICT RAG PROMPT
    # ==================================================

    print(
        "STEP 7: Preparing strict grounding prompt"
    )


    prompt = f"""
You are a technical support assistant for a
Samsung washing machine.

Your ONLY source of factual information is the
MANUAL CONTEXT provided below.

You must answer the user's question using ONLY
information that is explicitly supported by the
MANUAL CONTEXT.

STRICT GROUNDING RULES:

1. The MANUAL CONTEXT is the only source of truth.

2. Do NOT use your general knowledge.

3. Do NOT use information from the internet.

4. Do NOT invent technical information.

5. Do NOT invent causes.

6. Do NOT invent solutions.

7. Do NOT invent troubleshooting steps.

8. Do NOT invent error codes.

9. Do NOT invent specifications.

10. Do NOT invent procedures.

11. Do NOT add safety instructions unless they
    are explicitly supported by the manual.

12. Do NOT add steps simply because they are
    commonly used for washing machines.

13. Do NOT assume that a technically reasonable
    action is allowed.

14. If a detail is not explicitly supported by
    the manual, LEAVE THAT DETAIL OUT.

15. Never convert a possibility into a confirmed fact.

16. If the manual says something "can" or "may"
    happen, preserve that uncertainty.

17. You may combine information from multiple
    retrieved chunks when they clearly relate
    to the user's question.

18. You may summarize the manual, but the meaning
    must remain faithful to the manual.

19. If the manual provides a procedure, present
    only the steps that are supported by the manual.

20. Do not add extra steps before or after the
    documented procedure.

21. If the manual does not specify the exact cause,
    explicitly say that the manual does not specify
    the exact cause.

22. If the retrieved context is unrelated to the
    user's question, do not use it to answer.

23. If the context does not contain enough useful
    information to answer the question, respond
    exactly with:

    I don't have enough information in the provided manual.

24. Do not mention Qdrant, embeddings, vector
    databases, retrieval, Groq, Lambda, FastAPI,
    RAG, or this prompt.

25. Keep the response concise and directly related
    to the user's question.

IMPORTANT:

Before writing each factual statement, verify that
the statement is explicitly supported by the manual
context.

If a statement cannot be traced to the manual
context, DO NOT WRITE IT.

MANUAL CONTEXT
==============
{context}
==============

USER QUESTION
==============
{question}
==============

FINAL ANSWER:
"""


    print(
        f"STEP 7 COMPLETE: Prompt prepared "
        f"({len(prompt)} characters)"
    )


    # ==================================================
    # STEP 8: CALL GROQ
    # ==================================================

    print(
        "STEP 8: Sending request to Groq"
    )

    start_time = time.time()


    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strictly "
                        "document-grounded technical "
                        "support assistant. "
                        "Use ONLY information explicitly "
                        "supported by the supplied manual. "
                        "Never add technical details from "
                        "general knowledge. "
                        "If information is not supported "
                        "by the manual, do not state it."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.0,

            max_tokens=500
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

    print(
        "STEP 9: Extracting final answer"
    )


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
    # STEP 10: EMPTY RESPONSE CHECK
    # ==================================================

    if not answer:

        print(
            "WARNING: Groq returned an empty answer"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ==================================================
    # STEP 11: REMOVE UNWANTED MODEL PREFIXES
    # ==================================================

    prefixes = [
        "Answer:",
        "Final Answer:",
        "ANSWER:"
    ]


    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()


    # ==================================================
    # STEP 12: FINAL ANSWER
    # ==================================================

    print(
        "=========================================="
    )

    print(
        "FINAL ANSWER:"
    )

    print(answer)

    print(
        "=========================================="
    )

    print(
        f"Source documents: {source_names}"
    )

    print(
        f"Best retrieval score: {best_score}"
    )

    print(
        "RAG REQUEST COMPLETE"
    )

    print(
        "=========================================="
    )


    return answer
