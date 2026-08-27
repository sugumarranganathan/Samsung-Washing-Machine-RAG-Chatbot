FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

# Install Python dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Create model directory
RUN mkdir -p /opt/models/all-MiniLM-L6-v2

# Download embedding model during Docker build
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='sentence-transformers/all-MiniLM-L6-v2', local_dir='/opt/models/all-MiniLM-L6-v2')"

# Lambda handler
CMD ["app.main.handler"]
