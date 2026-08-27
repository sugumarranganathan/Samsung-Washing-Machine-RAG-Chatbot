FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}


# --------------------------------------------------
# Install Python dependencies
# --------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# --------------------------------------------------
# Hugging Face configuration
# --------------------------------------------------

ENV HF_HOME=/opt/huggingface
ENV TRANSFORMERS_CACHE=/opt/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/opt/huggingface/sentence_transformers

# Prevent Hugging Face from trying to download
# the model when Lambda is running
ENV HF_HUB_OFFLINE=1


# --------------------------------------------------
# Copy application
# --------------------------------------------------

COPY app/ ./app/


# --------------------------------------------------
# Download Sentence Transformer during Docker build
# --------------------------------------------------

RUN mkdir -p /opt/models && \
    python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('/opt/models/all-MiniLM-L6-v2')"


# --------------------------------------------------
# Lambda handler
# --------------------------------------------------

CMD ["app.main.handler"]
