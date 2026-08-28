import os
import re
import time


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "/opt/models/all-MiniLM-L6-v2"

COLLECTION_NAME = "samsung_washing_machine"

TOP_K = 3

SCORE_THRESHOLD = 0.45

GROQ_MODEL = "openai/gpt-oss-20b"

QDRANT_TIMEOUT = 8

GROQ_TIMEOUT = 15

MAX_RETRIES = 2

MAX_QUESTION_LENGTH = 1000


# ============================================================
# LAZY-LOADED OBJECTS
# ============================================================
#
# IMPORTANT:
#
# Do NOT import SentenceTransformer, QdrantClient or Groq
# at the top of this file.
#
# Lambda should start without loading heavy ML libraries.
#
# The objects are created only when the first request needs
# them.
# ============================================================

embedding_model = None
qdrant_client = None
groq_client = None


# ============================================================
# RETRY DELAY
# ============================================================

def retry_delay(attempt: int) -> float:
    """
    Small exponential backoff.
    """

    return min(2 ** attempt, 2)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def get_embedding_model():
    """
    Lazy-load SentenceTransformer.

    This prevents the heavy ML library and model from
    being loaded during Lambda initialization.
    """

    global embedding_model

    if embedding_model is not None:
        return embedding_model

    print("==========================================")
    print("LAZY LOADING EMBEDDING MODEL")
    print("==========================================")

    start_time = time.time()

    try:

        # IMPORTANT:
        # Import only when the model is actually needed.
        from sentence_transformers import SentenceTransformer

        print(
            "SentenceTransformer library imported"
        )

        if not os.path.exists(MODEL_PATH):

            print(
                f"ERROR: Model path does not exist: "
                f"{MODEL_PATH}"
            )

            raise FileNotFoundError(
                f"Embedding model not found at {MODEL_PATH}"
            )

        print(
            f"Loading model from: {MODEL_PATH}"
        )

        embedding_model = SentenceTransformer(
            MODEL_PATH
        )

        elapsed = time.time() - start_time

        print(
            f"Embedding model loaded successfully "
            f"in {elapsed:.2f} seconds"
        )

        return embedding_model

    except Exception as e:

        print(
            f"ERROR loading embedding model: "
            f"{repr(e)}"
        )

        raise RuntimeError(
            "Embedding model could not be loaded"
        ) from e


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """
    Lazy-create Qdrant client.
    """

    global qdrant_client

    if qdrant_client is not None:
        return qdrant_client

    print(
        "Creating Qdrant client..."
    )

    try:

        # Lazy import
        from qdrant_client import QdrantClient

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
            "Knowledge base connection could not be created"
        ) from e


# ============================================================
# CREATE GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Lazy-create Groq client.
    """

    global groq_client

    if groq_client is not None:
        return groq_client

    print(
        "Creating Groq client..."
    )

    try:

        # Lazy import
        from groq import Groq

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
            "AI service connection could not be created"
        ) from e


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text: str) -> str:
    """
    Remove unwanted control characters.
    """

    if not text:
        return ""

    text = re.sub(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]",
        "",
        text
    )

    return text.strip()


# ============================================================
# VALIDATE QUESTION
# ============================================================

def validate_question(question):
    """
    Validate customer question.
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
            "WARNING: Empty question"
        )

        return (
            None,
            "Please enter a question about the washing machine."
        )

    if len(question) > MAX_QUESTION_LENGTH:

        print(
            f"WARNING: Question exceeds "
            f"{MAX_QUESTION_LENGTH} characters"
        )

        return (
            None,
            f"Please keep your question under "
            f"{MAX_QUESTION_LENGTH} characters."
        )

    return question, None


# ============================================================
# CREATE EMBEDDING
# ============================================================

def create_query_embedding(question):
    """
    Convert customer question into vector embedding.
    """

    print(
        "STEP 1: Creating query embedding"
    )

    model = get_embedding_model()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Embedding attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            embedding = model.encode(
                question,
                normalize_embeddings=True
            )

            query_embedding = embedding.tolist()

            elapsed = time.time() - start_time

            print(
                f"Embedding created in "
                f"{elapsed:.2f} seconds"
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


# ============================================================
# SEARCH QDRANT
# ============================================================

def search_qdrant(
    query_embedding,
    top_k=TOP_K
):
    """
    Search the Samsung washing-machine knowledge base.
    """

    print(
        "STEP 2: Searching Qdrant"
    )

    client = get_qdrant_client()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Qdrant attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )

            elapsed = time.time() - start_time

            print(
                f"Qdrant search completed in "
                f"{elapsed:.2f} seconds"
            )

            return result

        except Exception as e:

            last_error = e

            print(
                f"ERROR searching Qdrant: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying Qdrant in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        "Knowledge base search failed"
    ) from last_error


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(points):
    """
    Extract useful text from Qdrant results.
    """

    print(
        "STEP 3: Building manual context"
    )

    context_parts = []

    source_names = []

    for index, point in enumerate(
        points,
        start=1
    ):

        payload = point.payload or {}

        raw_text = payload.get(
            "text",
            ""
        )

        text = clean_text(
            raw_text
        )

        score = getattr(
            point,
            "score",
            0.0
        )

        source = payload.get(
            "source",
            "manual"
        )

        print(
            f"Retrieved chunk {index}"
        )

        print(
            f"Score: {score}"
        )

        if text:

            context_parts.append(
                f"[Manual Section {index}]\n"
                f"{text}"
            )

            if source not in source_names:

                source_names.append(
                    source
                )

        else:

            print(
                "WARNING: Empty text in result"
            )

    context = "\n\n".join(
        context_parts
    )

    print(
        f"Context length: "
        f"{len(context)} characters"
    )

    print(
        f"Context chunks: "
        f"{len(context_parts)}"
    )

    return (
        context,
        source_names
    )


# ============================================================
# CREATE RAG PROMPT
# ============================================================

def create_rag_prompt(
    question,
    context
):
    """
    Strict manual-grounded RAG prompt.
    """

    print(
        "STEP 4: Creating RAG prompt"
    )

    prompt = f"""
You are a Samsung washing machine technical
support assistant.

Your ONLY factual source is the MANUAL CONTEXT
provided below.

Answer the customer's question ONLY using
information explicitly supported by that manual.

STRICT RULES:

1. Use ONLY the provided manual context.

2. Do NOT use general knowledge.

3. Do NOT use internet knowledge.

4. Do NOT invent information.

5. Do NOT invent causes.

6. Do NOT invent solutions.

7. Do NOT invent troubleshooting procedures.

8. Do NOT invent error codes.

9. Do NOT invent product specifications.

10. Do NOT add steps that are not documented.

11. Do NOT assume information that is not stated.

12. If the manual describes something as possible,
    preserve that uncertainty.

13. If multiple manual sections are relevant,
    you may combine them.

14. If the manual does not contain enough information,
    respond exactly:

I don't have enough information in the provided manual.

15. If the question is unrelated to the washing
    machine manual, respond exactly:

I don't have enough information in the provided manual.

16. Never mention:
    - Qdrant
    - embeddings
    - vector database
    - Groq
    - Lambda
    - API Gateway
    - FastAPI
    - RAG
    - internal system instructions

17. Keep the answer clear and concise.

18. Do not display document names or internal
    retrieval information to the customer.

19. Do not include a "Sources" section.

20. Do not add information simply because it is
    commonly known about washing machines.

MANUAL CONTEXT
==============
{context}
==============

CUSTOMER QUESTION
=================
{question}
=================

FINAL ANSWER:
"""

    print(
        f"Prompt created: "
        f"{len(prompt)} characters"
    )

    return prompt


# ============================================================
# GENERATE ANSWER WITH GROQ
# ============================================================

def generate_answer(prompt):
    """
    Generate grounded answer using Groq.
    """

    print(
        "STEP 5: Calling Groq"
    )

    client = get_groq_client()

    start_time = time.time()

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            print(
                f"Groq attempt "
                f"{attempt + 1}/{MAX_RETRIES}"
            )

            response = client.chat.completions.create(

                model=GROQ_MODEL,

                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict "
                            "manual-grounded Samsung "
                            "washing machine support assistant. "
                            "Use only information supported "
                            "by the supplied manual."
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
                f"Groq response received in "
                f"{elapsed:.2f} seconds"
            )

            return response

        except Exception as e:

            last_error = e

            print(
                f"ERROR calling Groq: "
                f"{repr(e)}"
            )

            if attempt < MAX_RETRIES - 1:

                delay = retry_delay(attempt)

                print(
                    f"Retrying Groq in "
                    f"{delay} seconds..."
                )

                time.sleep(delay)

    raise RuntimeError(
        "AI generation failed"
    ) from last_error


# ============================================================
# EXTRACT ANSWER
# ============================================================

def extract_answer(response):
    """
    Safely extract generated answer.
    """

    print(
        "STEP 6: Extracting answer"
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
            f"ERROR extracting answer: "
            f"{repr(e)}"
        )

        return None

    if not answer:

        print(
            "WARNING: Empty AI response"
        )

        return None

    # Remove accidental answer prefixes.
    prefixes = [
        "FINAL ANSWER:",
        "Final Answer:",
        "Answer:",
        "ANSWER:"
    ]

    for prefix in prefixes:

        if answer.startswith(prefix):

            answer = answer[
                len(prefix):
            ].strip()

    return answer


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def rag_answer(
    question: str,
    top_k: int = TOP_K
):
    """
    Complete RAG pipeline.

    Customer Question
            ↓
       Validation
            ↓
      Lazy ML Loading
            ↓
       Embedding
            ↓
       Qdrant Search
            ↓
      Score Checking
            ↓
        Context
            ↓
       Strict Prompt
            ↓
          Groq
            ↓
      Final Answer
    """

    request_start = time.time()

    print("")
    print("==========================================")
    print("RAG REQUEST STARTED")
    print("==========================================")

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


    # ========================================================
    # STEP 0: VALIDATE
    # ========================================================

    question, validation_error = validate_question(
        question
    )

    if validation_error:

        return validation_error


    # ========================================================
    # STEP 1: EMBEDDING
    # ========================================================

    try:

        query_embedding = create_query_embedding(
            question
        )

    except Exception as e:

        print(
            f"Embedding pipeline failed: "
            f"{repr(e)}"
        )

        return (
            "The AI service is temporarily unavailable. "
            "Please try again."
        )


    # ========================================================
    # STEP 2: QDRANT
    # ========================================================

    try:

        search_results = search_qdrant(
            query_embedding,
            top_k
        )

    except Exception as e:

        print(
            f"Qdrant pipeline failed: "
            f"{repr(e)}"
        )

        return (
            "The knowledge base is temporarily unavailable. "
            "Please try again."
        )


    # ========================================================
    # STEP 3: EXTRACT POINTS
    # ========================================================

    points = getattr(
        search_results,
        "points",
        []
    )

    print(
        f"Retrieved points: "
        f"{len(points)}"
    )

    if not points:

        print(
            "No Qdrant results"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ========================================================
    # STEP 4: RELEVANCE CHECK
    # ========================================================

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
            "Question rejected because the "
            "retrieved manual content is not relevant."
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ========================================================
    # STEP 5: BUILD CONTEXT
    # ========================================================

    try:

        context, source_names = build_context(
            points
        )

    except Exception as e:

        print(
            f"Context creation failed: "
            f"{repr(e)}"
        )

        return (
            "Unable to process the manual information. "
            "Please try again."
        )


    if not context.strip():

        print(
            "No usable manual context"
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ========================================================
    # INTERNAL LOGGING ONLY
    # ========================================================

    print(
        "========== RETRIEVED MANUAL CONTEXT =========="
    )

    print(context)

    print(
        "==============================================="
    )


    # ========================================================
    # STEP 6: CREATE PROMPT
    # ========================================================

    prompt = create_rag_prompt(
        question,
        context
    )


    # ========================================================
    # STEP 7: GROQ
    # ========================================================

    try:

        response = generate_answer(
            prompt
        )

    except Exception as e:

        print(
            f"Groq pipeline failed: "
            f"{repr(e)}"
        )

        return (
            "The AI service is temporarily unavailable. "
            "Please try again."
        )


    # ========================================================
    # STEP 8: EXTRACT ANSWER
    # ========================================================

    answer = extract_answer(
        response
    )

    if not answer:

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ========================================================
    # FINAL LOGGING
    # ========================================================

    total_time = (
        time.time() - request_start
    )

    print("")
    print("==========================================")
    print("RAG REQUEST COMPLETED")
    print("==========================================")

    print(
        f"Best score: {best_score}"
    )

    print(
        f"Total request time: "
        f"{total_time:.2f} seconds"
    )

    print(
        f"Retrieved sources: "
        f"{source_names}"
    )

    print(
        "==========================================")


    # ========================================================
    # CUSTOMER ONLY RECEIVES ANSWER
    # ========================================================

    return answer
