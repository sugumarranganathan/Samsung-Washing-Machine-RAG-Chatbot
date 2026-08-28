import os
import time
import re

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq


# =========================================================
# ENVIRONMENT VARIABLES
# =========================================================

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "/opt/models/all-MiniLM-L6-v2"

COLLECTION_NAME = "samsung_washing_machine"

# Number of chunks retrieved from Qdrant
TOP_K = 3

# Minimum similarity score
SCORE_THRESHOLD = 0.45

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"

# Timeouts
QDRANT_TIMEOUT = 10
GROQ_TIMEOUT = 20

# Retry configuration
MAX_RETRIES = 2

# Maximum question size
MAX_QUESTION_LENGTH = 1000


# =========================================================
# GLOBAL CLIENTS / MODEL
# =========================================================
#
# IMPORTANT:
# Do NOT load SentenceTransformer during Lambda initialization.
#
# The old code did:
#
# embedding_model = SentenceTransformer(MODEL_PATH)
#
# That caused Lambda INIT timeout.
#
# We now load the model only when the first request arrives.
# =========================================================

embedding_model = None
qdrant_client = None
groq_client = None


# =========================================================
# RETRY HELPER
# =========================================================

def retry_delay(attempt: int) -> float:
    """
    Exponential backoff delay.

    attempt 0 -> 1 second
    attempt 1 -> 2 seconds
    """
    return min(2 ** attempt, 4)


# =========================================================
# LOAD EMBEDDING MODEL LAZILY
# =========================================================

def get_embedding_model():
    """
    Load the SentenceTransformer model only when required.

    This prevents the model from being loaded during
    Lambda cold-start initialization.
    """

    global embedding_model

    if embedding_model is not None:
        return embedding_model

    print("==========================================")
    print("LOADING EMBEDDING MODEL")
    print("==========================================")

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Embedding model load attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            embedding_model = SentenceTransformer(
                MODEL_PATH
            )

            elapsed = time.time() - start_time

            print(
                f"EMBEDDING MODEL LOADED "
                f"in {elapsed:.2f} seconds"
            )

            return embedding_model

        except Exception as e:

            last_error = e

            print(
                f"ERROR loading embedding model: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying model load in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    print(
        "FATAL: Unable to load embedding model"
    )

    raise RuntimeError(
        "Unable to load embedding model"
    ) from last_error


# =========================================================
# CREATE QDRANT CLIENT LAZILY
# =========================================================

def get_qdrant_client():
    """
    Create Qdrant client only when required.
    """

    global qdrant_client

    if qdrant_client is not None:
        return qdrant_client

    print("Creating Qdrant client...")

    try:

        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=QDRANT_TIMEOUT
        )

        print(
            "Qdrant client created successfully"
        )

        return qdrant_client

    except Exception as e:

        print(
            f"ERROR creating Qdrant client: "
            f"{repr(e)}"
        )

        raise RuntimeError(
            "Unable to create knowledge-base client"
        ) from e


# =========================================================
# CREATE GROQ CLIENT LAZILY
# =========================================================

def get_groq_client():
    """
    Create Groq client only when required.
    """

    global groq_client

    if groq_client is not None:
        return groq_client

    print("Creating Groq client...")

    try:

        groq_client = Groq(
            api_key=GROQ_API_KEY,
            timeout=GROQ_TIMEOUT,
            max_retries=0
        )

        print(
            "Groq client created successfully"
        )

        return groq_client

    except Exception as e:

        print(
            f"ERROR creating Groq client: "
            f"{repr(e)}"
        )

        raise RuntimeError(
            "Unable to create AI service client"
        ) from e


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:
    """
    Remove unwanted control characters while
    preserving normal whitespace.
    """

    if not text:
        return ""

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    return text.strip()


# =========================================================
# VALIDATE QUESTION
# =========================================================

def validate_question(question):
    """
    Validate and normalize user question.
    """

    if not isinstance(question, str):

        print(
            "WARNING: Question is not a string"
        )

        return (
            None,
            "Please provide a valid question."
        )

    question = question.strip()

    if not question:

        print(
            "WARNING: Empty question received"
        )

        return (
            None,
            "Please enter a question about the washing machine."
        )

    if len(question) > MAX_QUESTION_LENGTH:

        print(
            f"WARNING: Question exceeded "
            f"{MAX_QUESTION_LENGTH} characters"
        )

        return (
            None,
            f"Please keep your question under "
            f"{MAX_QUESTION_LENGTH} characters."
        )

    return question, None


# =========================================================
# CREATE QUERY EMBEDDING WITH RETRY
# =========================================================

def create_query_embedding(question):
    """
    Create query embedding with retry handling.
    """

    print("STEP 4: Creating query embedding")

    model = get_embedding_model()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Embedding attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            query_embedding = model.encode(
                question,
                normalize_embeddings=True
            ).tolist()

            elapsed = time.time() - start_time

            print(
                f"STEP 4 COMPLETE: Embedding created "
                f"in {elapsed:.2f} seconds"
            )

            print(
                f"Embedding dimensions: "
                f"{len(query_embedding)}"
            )

            return query_embedding

        except Exception as e:

            last_error = e

            print(
                f"ERROR creating embedding: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying embedding in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        "Unable to create query embedding"
    ) from last_error


# =========================================================
# SEARCH QDRANT WITH RETRY
# =========================================================

def search_qdrant(query_embedding, top_k):
    """
    Search Qdrant with retry handling.
    """

    print("STEP 5: Searching Qdrant")

    client = get_qdrant_client()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Qdrant search attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            search_results = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )

            elapsed = time.time() - start_time

            print(
                f"STEP 5 COMPLETE: Qdrant search "
                f"completed in {elapsed:.2f} seconds"
            )

            return search_results

        except Exception as e:

            last_error = e

            print(
                f"ERROR in Qdrant: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying Qdrant search in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        "Unable to search the knowledge base"
    ) from last_error


# =========================================================
# BUILD CONTEXT
# =========================================================

def build_context(points):
    """
    Convert retrieved Qdrant points into
    clean text context.
    """

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
                f"[Document Chunk {index}]\n"
                f"{text}"
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

    return context, source_names


# =========================================================
# CREATE STRICT RAG PROMPT
# =========================================================

def create_rag_prompt(
    question,
    context
):
    """
    Create a strict document-grounded prompt.
    """

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

    return prompt


# =========================================================
# CALL GROQ WITH RETRY
# =========================================================

def generate_answer(prompt):
    """
    Send prompt to Groq with controlled retry handling.
    """

    print(
        "STEP 8: Sending request to Groq"
    )

    client = get_groq_client()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Groq request attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            response = client.chat.completions.create(

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

            elapsed = time.time() - start_time

            print(
                f"STEP 8 COMPLETE: Groq response "
                f"received in {elapsed:.2f} seconds"
            )

            return response

        except Exception as e:

            last_error = e

            print(
                f"ERROR in Groq: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying Groq request in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        "Unable to generate AI response"
    ) from last_error


# =========================================================
# EXTRACT ANSWER
# =========================================================

def extract_answer(response):
    """
    Safely extract answer from Groq response.
    """

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

        return None

    if not answer:

        print(
            "WARNING: Groq returned an empty answer"
        )

        return None

    # Remove unwanted prefixes
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

    return answer


# =========================================================
# MAIN RAG FUNCTION
# =========================================================

def rag_answer(
    question: str,
    top_k: int = TOP_K
):
    """
    Complete RAG pipeline.

    Pipeline:

    User Question
        ↓
    Validation
        ↓
    Lazy Model Loading
        ↓
    Query Embedding
        ↓
    Qdrant Search
        ↓
    Relevance Check
        ↓
    Context Building
        ↓
    Strict RAG Prompt
        ↓
    Groq
        ↓
    Answer Extraction
        ↓
    Final Answer
    """

    print("")
    print("==========================================")
    print("RAG REQUEST STARTED")
    print("==========================================")

    request_start_time = time.time()

    print(
        f"Question: {question}"
    )

    print(
        f"Top K: {top_k}"
    )

    print(
        f"Score threshold: "
        f"{SCORE_THRESHOLD}"
    )

    print("==========================================")


    # =====================================================
    # STEP 1: VALIDATE QUESTION
    # =====================================================

    question, validation_error = validate_question(
        question
    )

    if validation_error:

        return validation_error


    # =====================================================
    # STEP 2: CREATE QUERY EMBEDDING
    # =====================================================

    try:

        query_embedding = create_query_embedding(
            question
        )

    except Exception as e:

        print(
            f"FATAL ERROR during embedding: "
            f"{repr(e)}"
        )

        return (
            "The AI service is temporarily unavailable. "
            "Please try again."
        )


    # =====================================================
    # STEP 3: SEARCH QDRANT
    # =====================================================

    try:

        search_results = search_qdrant(
            query_embedding,
            top_k
        )

    except Exception as e:

        print(
            f"FATAL ERROR during Qdrant search: "
            f"{repr(e)}"
        )

        return (
            "The knowledge base is temporarily "
            "unavailable. Please try again."
        )


    # =====================================================
    # STEP 4: CHECK SEARCH RESULTS
    # =====================================================

    points = getattr(
        search_results,
        "points",
        []
    )

    print(
        f"Number of results: "
        f"{len(points)}"
    )

    if not points:

        print(
            "WARNING: Qdrant returned no results"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # =====================================================
    # STEP 5: CHECK RELEVANCE SCORE
    # =====================================================

    best_score = getattr(
        points[0],
        "score",
        0.0
    )

    if best_score is None:

        best_score = 0.0


    print(
        f"Best similarity score: "
        f"{best_score}"
    )


    if best_score < SCORE_THRESHOLD:

        print(
            "WARNING: Best result is below "
            "the relevance threshold."
        )

        print(
            f"Best score: {best_score}"
        )

        print(
            f"Required score: "
            f"{SCORE_THRESHOLD}"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # =====================================================
    # STEP 6: BUILD CONTEXT
    # =====================================================

    try:

        context, source_names = build_context(
            points
        )

    except Exception as e:

        print(
            f"ERROR building context: "
            f"{repr(e)}"
        )

        return (
            "Unable to process the knowledge-base "
            "information. Please try again."
        )


    # =====================================================
    # STEP 7: CHECK CONTEXT
    # =====================================================

    if not context.strip():

        print(
            "WARNING: Qdrant returned no usable "
            "text context"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # =====================================================
    # DISPLAY RETRIEVED CONTEXT
    # =====================================================

    print(
        "========== RETRIEVED CONTEXT =========="
    )

    print(context)

    print(
        "========================================"
    )


    # =====================================================
    # STEP 8: CREATE STRICT RAG PROMPT
    # =====================================================

    prompt = create_rag_prompt(
        question,
        context
    )


    # =====================================================
    # STEP 9: CALL GROQ
    # =====================================================

    try:

        response = generate_answer(
            prompt
        )

    except Exception as e:

        print(
            f"FATAL ERROR during Groq generation: "
            f"{repr(e)}"
        )

        return (
            "The AI service is temporarily unavailable. "
            "Please try again."
        )


    # =====================================================
    # STEP 10: EXTRACT ANSWER
    # =====================================================

    answer = extract_answer(
        response
    )

    if not answer:

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # =====================================================
    # STEP 11: FINAL LOGGING
    # =====================================================

    total_time = (
        time.time() - request_start_time
    )

    print("")
    print("==========================================")
    print("FINAL ANSWER")
    print("==========================================")

    print(answer)

    print("==========================================")

    print(
        f"Source documents: "
        f"{source_names}"
    )

    print(
        f"Best retrieval score: "
        f"{best_score}"
    )

    print(
        f"Total request time: "
        f"{total_time:.2f} seconds"
    )

    print(
        "RAG REQUEST COMPLETE"
    )

    print(
        "=========================================="
    )

    return answer
