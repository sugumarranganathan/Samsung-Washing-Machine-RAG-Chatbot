FROM public.ecr.aws/lambda/python:3.12

WORKDIR ${LAMBDA_TASK_ROOT}

# Install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ ./app/

# Download the Sentence Transformer model
RUN mkdir -p /opt/models/all-MiniLM-L6-v2

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='sentence-transformers/all-MiniLM-L6-v2', \
    local_dir='/opt/models/all-MiniLM-L6-v2', \
    allow_patterns=[ \
        'config.json', \
        'config_sentence_transformers.json', \
        'data_config.json', \
        'modules.json', \
        'sentence_bert_config.json', \
        'model.safetensors', \
        'tokenizer.json', \
        'tokenizer_config.json', \
        'special_tokens_map.json', \
        '1_Pooling/*' \
    ] \
)"

# VERIFY that the model was really downloaded
RUN test -f /opt/models/all-MiniLM-L6-v2/model.safetensors

# Show model files in GitHub Actions build log
RUN ls -lh /opt/models/all-MiniLM-L6-v2/

# Lambda handler
CMD ["app.main.handler"]
