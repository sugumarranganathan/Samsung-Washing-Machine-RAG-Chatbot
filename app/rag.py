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

# Retrieve more candidates so natural customer wording has
# a better chance of finding the correct manual section.
TOP_K = 8

# Semantic threshold for clearly washing-machine-related
# questions.
DOMAIN_SCORE_THRESHOLD = 0.30

# Stronger threshold for questions without obvious
# washing-machine/domain signals.
GENERAL_SCORE_THRESHOLD = 0.45

# When lexical evidence is found in retrieved manual text,
# allow the result even if semantic score is slightly lower.
LEXICAL_SCORE_THRESHOLD = 0.15

GROQ_MODEL = "openai/gpt-oss-20b"

QDRANT_TIMEOUT = 8

GROQ_TIMEOUT = 15

MAX_RETRIES = 2

MAX_QUESTION_LENGTH = 1000

# Number of retrieved chunks supplied to the LLM.
MAX_CONTEXT_CHUNKS = 5

# Maximum characters per individual manual chunk.
MAX_CHUNK_LENGTH = 5000


# ============================================================
# LAZY-LOADED OBJECTS
# ============================================================

embedding_model = None
qdrant_client = None
groq_client = None


# ============================================================
# COMMON WASHING-MACHINE TERMS
# ============================================================

WASHING_MACHINE_TERMS = {
    "washing",
    "washer",
    "machine",
    "laundry",
    "clothes",
    "clothing",
    "drum",
    "cycle",
    "wash",
    "rinse",
    "spin",
    "detergent",
    "door",
    "lock",
    "locked",
    "unlock",
    "water",
    "drain",
    "draining",
    "drainage",
    "hose",
    "filter",
    "vibration",
    "vibrating",
    "vibrate",
    "shaking",
    "shake",
    "noise",
    "noisy",
    "sound",
    "error",
    "code",
    "start",
    "starting",
    "run",
    "running",
    "operate",
    "operation",
    "clean",
    "cleaning",
    "maintenance",
    "leak",
    "leaking",
    "temperature",
    "spin",
    "load",
}


# ============================================================
# DOMAIN / INTENT TERMS
# ============================================================

DOMAIN_PHRASES = [
    "washing machine",
    "washing-machine",
    "washer",
    "laundry machine",
    "wash cycle",
    "spin cycle",
    "rinse cycle",
    "wash program",
    "washing cycle",
    "machine door",
    "washer door",
    "drain hose",
    "drain filter",
    "error code",
]


# ============================================================
# SYMPTOM / TOPIC GROUPS
# ============================================================

TOPIC_GROUPS = {
    "vibration": {
        "vibration",
        "vibrating",
        "vibrate",
        "shaking",
        "shake",
        "shakes",
        "wobble",
        "wobbling",
        "unstable",
    },

    "noise": {
        "noise",
        "noisy",
        "sound",
        "sounds",
        "loud",
        "rattling",
        "rattle",
        "banging",
        "bang",
        "humming",
        "hum",
    },

    "door": {
        "door",
        "locked",
        "lock",
        "unlock",
        "unlocked",
        "open",
        "opened",
        "opening",
        "close",
        "closed",
    },

    "water_supply": {
        "water",
        "supply",
        "inlet",
        "inlet-hose",
        "tap",
        "faucet",
        "fill",
        "filling",
    },

    "drain": {
        "drain",
        "draining",
        "drainage",
        "drainage",
        "hose",
        "filter",
        "pump",
        "water-out",
        "empty",
    },

    "start_run": {
        "start",
        "starting",
        "run",
        "running",
        "work",
        "working",
        "operate",
        "operating",
        "turn",
        "turning",
        "begin",
        "beginning",
    },

    "cycle": {
        "cycle",
        "wash",
        "washing",
        "rinse",
        "spin",
        "program",
        "mode",
        "setting",
        "settings",
    },

    "maintenance": {
        "clean",
        "cleaning",
        "maintenance",
        "maintain",
        "filter",
        "care",
        "service",
    },

    "error": {
        "error",
        "code",
        "fault",
        "display",
        "message",
        "warning",
    },
}


# ============================================================
# ERROR CODE DETECTION
# ============================================================

def extract_error_codes(question: str):
    """
    Detect common washing-machine error-code patterns.

    This function does NOT invent the meaning of an error code.
    It only detects the code so Qdrant can search the manual
    more effectively.
    """

    if not question:
        return []

    normalized = question.upper()

    patterns = [
        r"\b\d+[A-Z]\b",
        r"\b[A-Z]{1,3}\d{1,3}\b",
        r"\b[A-Z]{1,3}\b(?=\s+ERROR\b)",
        r"\bERROR\s+CODE\s+[A-Z0-9]+\b",
    ]

    found = []

    for pattern in patterns:
        matches = re.findall(pattern, normalized)

        for match in matches:

            code = match.strip()

            if code not in found:
                found.append(code)

    return found


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

    Heavy ML dependencies are loaded only when needed.
    """

    global embedding_model

    if embedding_model is not None:
        return embedding_model

    print("==========================================")
    print("LAZY LOADING EMBEDDING MODEL")
    print("==========================================")

    start_time = time.time()

    try:

        from sentence_transformers import SentenceTransformer

        print("SentenceTransformer library imported")

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

    print("Creating Qdrant client...")

    try:

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

    print("Creating Groq client...")

    try:

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
# NORMALIZE QUERY
# ============================================================

def normalize_query(question: str) -> str:
    """
    Normalize natural customer wording.

    This does not add factual information.
    It only creates search-friendly wording.
    """

    text = question.lower().strip()

    replacements = {
        "washing-machine": "washing machine",
        "washer": "washing machine",
        "washingmachine": "washing machine",
        "not running": "not run",
        "won't run": "not run",
        "wont run": "not run",
        "doesn't run": "not run",
        "doesnt run": "not run",
        "not working": "not work",
        "won't work": "not work",
        "wont work": "not work",
        "doesn't work": "not work",
        "doesnt work": "not work",
        "not opened": "door not open",
        "door not opening": "door not open",
        "door won't open": "door not open",
        "door wont open": "door not open",
        "door doesn't open": "door not open",
        "door doesnt open": "door not open",
        "not going out": "not draining",
        "water not going out": "water not draining",
        "water staying": "water remains",
        "making noise": "noise",
        "noise is coming": "washing machine noise",
        "making a noise": "washing machine noise",
        "shaking badly": "washing machine shaking",
        "machine shaking": "washing machine shaking",
        "machine vibrating": "washing machine vibrating",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# DETECT DOMAIN INTENT
# ============================================================

def detect_domain_intent(question: str):
    """
    Determine whether the question appears related to
    washing-machine operation/support.

    This is intentionally broad so short natural-language
    customer questions are not incorrectly rejected.
    """

    normalized = normalize_query(question)

    # Strong explicit domain phrases.
    for phrase in DOMAIN_PHRASES:

        if phrase in normalized:
            return True

    words = set(
        re.findall(
            r"[a-z0-9]+",
            normalized.lower()
        )
    )

    # Any strong machine/support topic is enough.
    for topic_words in TOPIC_GROUPS.values():

        if words.intersection(topic_words):
            return True

    # Explicit error code.
    if extract_error_codes(question):
        return True

    return False


# ============================================================
# DETECT TOPICS
# ============================================================

def detect_topics(question: str):
    """
    Detect likely customer intent/topic.
    """

    normalized = normalize_query(question)

    words = set(
        re.findall(
            r"[a-z0-9]+",
            normalized.lower()
        )
    )

    topics = []

    for topic, topic_words in TOPIC_GROUPS.items():

        if words.intersection(topic_words):
            topics.append(topic)

    return topics


# ============================================================
# BUILD SEARCH QUERY
# ============================================================

def build_search_query(question: str) -> str:
    """
    Build a stronger semantic-search query.

    Important:
    This function does NOT supply factual answers.
    It only adds domain/search terminology.
    """

    normalized = normalize_query(question)

    topics = detect_topics(question)

    error_codes = extract_error_codes(question)

    additions = []

    # Make short customer questions more searchable.
    if detect_domain_intent(question):

        additions.append(
            "washing machine"
        )

    for topic in topics:

        if topic == "vibration":
            additions.append(
                "vibration shaking"
            )

        elif topic == "noise":
            additions.append(
                "noise sound"
            )

        elif topic == "door":
            additions.append(
                "door lock opening"
            )

        elif topic == "water_supply":
            additions.append(
                "water supply inlet"
            )

        elif topic == "drain":
            additions.append(
                "drain drainage hose filter"
            )

        elif topic == "start_run":
            additions.append(
                "machine start operation"
            )

        elif topic == "cycle":
            additions.append(
                "wash cycle operation"
            )

        elif topic == "maintenance":
            additions.append(
                "maintenance cleaning"
            )

        elif topic == "error":
            additions.append(
                "error code troubleshooting"
            )

    # Error code is retained exactly.
    if error_codes:

        additions.append(
            " ".join(error_codes)
        )

        additions.append(
            "washing machine error code troubleshooting"
        )

    search_query = (
        normalized
        + " "
        + " ".join(additions)
    )

    search_query = re.sub(
        r"\s+",
        " ",
        search_query
    ).strip()

    print(
        f"Original query: {question}"
    )

    print(
        f"Normalized query: {normalized}"
    )

    print(
        f"Search query: {search_query}"
    )

    print(
        f"Detected topics: {topics}"
    )

    print(
        f"Detected error codes: {error_codes}"
    )

    return search_query


# ============================================================
# VALIDATE QUESTION
# ============================================================

def validate_question(question):

    if not isinstance(question, str):

        print(
            "WARNING: Question is not a string"
        )

        return (
            None,
            "Please provide a valid question."
        )

    question = clean_text(question)

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
# LEXICAL MATCHING
# ============================================================

def lexical_match_score(
    question: str,
    text: str
) -> float:
    """
    Calculate lightweight lexical overlap.

    This is not used as the only retrieval method.
    It helps protect against semantic retrieval missing
    short phrases such as error codes.
    """

    if not question or not text:
        return 0.0

    question_lower = normalize_query(
        question
    ).lower()

    text_lower = text.lower()

    question_words = set(
        re.findall(
            r"[a-z0-9]+",
            question_lower
        )
    )

    text_words = set(
        re.findall(
            r"[a-z0-9]+",
            text_lower
        )
    )

    if not question_words or not text_words:
        return 0.0

    overlap = (
        question_words.intersection(
            text_words
        )
    )

    score = (
        len(overlap)
        / max(len(question_words), 1)
    )

    # Strong exact error-code evidence.
    error_codes = extract_error_codes(
        question
    )

    if error_codes:

        for code in error_codes:

            if code.lower() in text_lower:

                score = max(
                    score,
                    0.90
                )

    return min(score, 1.0)


# ============================================================
# SCORE RETRIEVED RESULTS
# ============================================================

def score_retrieved_points(
    question,
    points
):
    """
    Add lexical evidence to Qdrant semantic scores.

    Returns ranked candidates.
    """

    ranked = []

    domain_question = detect_domain_intent(
        question
    )

    error_codes = extract_error_codes(
        question
    )

    topics = detect_topics(question)

    print(
        f"Domain question: {domain_question}"
    )

    print(
        f"Topics: {topics}"
    )

    print(
        f"Error codes: {error_codes}"
    )

    for index, point in enumerate(
        points,
        start=1
    ):

        payload = point.payload or {}

        text = clean_text(
            payload.get("text", "")
        )

        semantic_score = getattr(
            point,
            "score",
            0.0
        )

        if semantic_score is None:
            semantic_score = 0.0

        lexical_score = lexical_match_score(
            question,
            text
        )

        combined_score = (
            (0.75 * float(semantic_score))
            +
            (0.25 * float(lexical_score))
        )

        exact_error_match = False

        if error_codes and text:

            text_lower = text.lower()

            exact_error_match = any(
                code.lower() in text_lower
                for code in error_codes
            )

        candidate = {
            "point": point,
            "text": text,
            "semantic_score": float(
                semantic_score
            ),
            "lexical_score": float(
                lexical_score
            ),
            "combined_score": float(
                combined_score
            ),
            "exact_error_match": (
                exact_error_match
            ),
            "domain_question": (
                domain_question
            ),
        }

        ranked.append(candidate)

        print(
            "------------------------------------------"
        )

        print(
            f"Candidate {index}"
        )

        print(
            f"Semantic score: "
            f"{semantic_score:.4f}"
        )

        print(
            f"Lexical score: "
            f"{lexical_score:.4f}"
        )

        print(
            f"Combined score: "
            f"{combined_score:.4f}"
        )

        print(
            f"Exact error match: "
            f"{exact_error_match}"
        )

    ranked.sort(
        key=lambda item: (
            item["exact_error_match"],
            item["combined_score"]
        ),
        reverse=True
    )

    return ranked


# ============================================================
# DETERMINE RELEVANCE
# ============================================================

def is_relevant(
    question,
    ranked_candidates
):
    """
    Determine whether retrieved manual content is useful.

    Rules:

    1. Exact error-code match is accepted.
    2. Clearly domain-related questions get a lower
       semantic threshold.
    3. Strong lexical evidence can rescue a slightly
       lower semantic result.
    4. Generic/unrelated questions retain the stronger
       threshold.
    """

    if not ranked_candidates:
        return False

    best = ranked_candidates[0]

    semantic_score = best[
        "semantic_score"
    ]

    lexical_score = best[
        "lexical_score"
    ]

    combined_score = best[
        "combined_score"
    ]

    exact_error_match = best[
        "exact_error_match"
    ]

    domain_question = best[
        "domain_question"
    ]

    print("==========================================")
    print("RELEVANCE EVALUATION")
    print("==========================================")

    print(
        f"Best semantic score: "
        f"{semantic_score:.4f}"
    )

    print(
        f"Best lexical score: "
        f"{lexical_score:.4f}"
    )

    print(
        f"Best combined score: "
        f"{combined_score:.4f}"
    )

    print(
        f"Domain question: "
        f"{domain_question}"
    )

    print(
        f"Exact error match: "
        f"{exact_error_match}"
    )

    # --------------------------------------------------------
    # RULE 1: Exact error-code match
    # --------------------------------------------------------

    if exact_error_match:

        print(
            "RELEVANCE ACCEPTED: "
            "exact error-code evidence"
        )

        return True

    # --------------------------------------------------------
    # RULE 2: Strong lexical evidence
    # --------------------------------------------------------

    if (
        domain_question
        and lexical_score
        >= LEXICAL_SCORE_THRESHOLD
        and combined_score
        >= 0.28
    ):

        print(
            "RELEVANCE ACCEPTED: "
            "domain + lexical evidence"
        )

        return True

    # --------------------------------------------------------
    # RULE 3: Domain question
    # --------------------------------------------------------

    if domain_question:

        if (
            semantic_score
            >= DOMAIN_SCORE_THRESHOLD
        ):

            print(
                "RELEVANCE ACCEPTED: "
                "domain semantic threshold"
            )

            return True

    # --------------------------------------------------------
    # RULE 4: Generic question
    # --------------------------------------------------------

    if (
        semantic_score
        >= GENERAL_SCORE_THRESHOLD
    ):

        print(
            "RELEVANCE ACCEPTED: "
            "general semantic threshold"
        )

        return True

    print(
        "RELEVANCE REJECTED"
    )

    return False


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    ranked_candidates,
    max_chunks=MAX_CONTEXT_CHUNKS
):
    """
    Build clean manual context.

    Only useful retrieved chunks are passed to the LLM.
    """

    print(
        "STEP 3: Building manual context"
    )

    context_parts = []

    source_names = []

    selected = ranked_candidates[
        :max_chunks
    ]

    for index, candidate in enumerate(
        selected,
        start=1
    ):

        text = candidate["text"]

        if not text:
            continue

        # Prevent one unusually large chunk from
        # consuming the entire prompt.
        if len(text) > MAX_CHUNK_LENGTH:

            text = text[
                :MAX_CHUNK_LENGTH
            ]

        source = candidate[
            "point"
        ].payload.get(
            "source",
            "manual"
        )

        semantic_score = candidate[
            "semantic_score"
        ]

        lexical_score = candidate[
            "lexical_score"
        ]

        print(
            f"Context chunk {index}"
        )

        print(
            f"Semantic score: "
            f"{semantic_score:.4f}"
        )

        print(
            f"Lexical score: "
            f"{lexical_score:.4f}"
        )

        context_parts.append(
            f"[Manual Section {index}]\n"
            f"{text}"
        )

        if source not in source_names:

            source_names.append(
                source
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

    print(
        "STEP 4: Creating RAG prompt"
    )

    prompt = f"""
You are a Samsung washing machine technical
support assistant.

Your ONLY factual source is the MANUAL CONTEXT
provided below.

The customer may use short, informal, incomplete,
misspelled, or conversational wording.

Interpret the customer's wording carefully, but
do not invent facts.

Answer the customer's question ONLY using
information supported by the manual context.

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
    combine them when appropriate.

14. If the manual does not contain enough information
    to answer the customer's question, respond exactly:

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
    - retrieval scores

17. Keep the answer clear and concise.

18. Do not display document names.

19. Do not display internal retrieval information.

20. Do not include a "Sources" section.

21. Do not add information simply because it is
    commonly known about washing machines.

22. If the manual gives several relevant troubleshooting
    points, provide all relevant documented points.

23. Prefer practical numbered steps or bullet points
    when the manual provides multiple actions.

24. If the customer uses informal wording such as
    "machine not run", "noise is coming", "door not opened",
    or "water not going out", understand the likely intent
    from the supplied manual context, but only state facts
    that the manual supports.

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
                            "by the supplied manual. "
                            "Never invent technical facts."
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
    Complete improved RAG pipeline.

    Customer Question
            ↓
       Validation
            ↓
      Query Normalization
            ↓
       Query Expansion
            ↓
        Embedding
            ↓
       Qdrant Search
            ↓
    Semantic + Lexical Ranking
            ↓
      Relevance Validation
            ↓
        Manual Context
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
        f"Domain threshold: "
        f"{DOMAIN_SCORE_THRESHOLD}"
    )

    print(
        f"General threshold: "
        f"{GENERAL_SCORE_THRESHOLD}"
    )

    print("==========================================")


    # ========================================================
    # STEP 0: VALIDATE
    # ========================================================

    question, validation_error = (
        validate_question(question)
    )

    if validation_error:

        return validation_error


    # ========================================================
    # STEP 0.5: BUILD SEARCH QUERY
    # ========================================================

    search_query = build_search_query(
        question
    )


    # ========================================================
    # STEP 1: EMBEDDING
    # ========================================================

    try:

        query_embedding = (
            create_query_embedding(
                search_query
            )
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
    # STEP 4: RANK RESULTS
    # ========================================================

    ranked_candidates = (
        score_retrieved_points(
            question,
            points
        )
    )


    # ========================================================
    # STEP 5: RELEVANCE CHECK
    # ========================================================

    relevant = is_relevant(
        question,
        ranked_candidates
    )

    if not relevant:

        print(
            "Question rejected because the "
            "retrieved manual content is not relevant."
        )

        return (
            "I don't have enough information "
            "in the provided manual."
        )


    # ========================================================
    # STEP 6: BUILD CONTEXT
    # ========================================================

    try:

        context, source_names = (
            build_context(
                ranked_candidates
            )
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
    # STEP 7: CREATE PROMPT
    # ========================================================

    prompt = create_rag_prompt(
        question,
        context
    )


    # ========================================================
    # STEP 8: GROQ
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
    # STEP 9: EXTRACT ANSWER
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
        time.time()
        - request_start
    )

    best = ranked_candidates[0]

    print("")
    print("==========================================")
    print("RAG REQUEST COMPLETED")
    print("==========================================")

    print(
        f"Best semantic score: "
        f"{best['semantic_score']:.4f}"
    )

    print(
        f"Best lexical score: "
        f"{best['lexical_score']:.4f}"
    )

    print(
        f"Best combined score: "
        f"{best['combined_score']:.4f}"
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
        "=========================================="
    )


    # ========================================================
    # CUSTOMER ONLY RECEIVES ANSWER
    # ========================================================

    return answer
