FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face cache during Docker build
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/huggingface/sentence_transformers

# Copy application
COPY app/ ./app/

# Create model directory
RUN mkdir -p /opt/models

# Download and save the Sentence Transformer model
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); model.save('/opt/models/all-MiniLM-L6-v2')"

# After the model is inside the image,
# prevent runtime downloads from Hugging Face
ENV HF_HUB_OFFLINE=1

CMD ["app.main.handler"]
