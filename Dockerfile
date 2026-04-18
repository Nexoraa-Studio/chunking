FROM python:3.11-slim-bookworm

# Runtime system deps for docling / opencv / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first so requirements.txt doesn't pull CUDA variants.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2,<3" "torchvision>=0.17,<1"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download all models into the HF cache inside the image so new container
# starts don't pay a 30-60s download on their first PDF upload.
# Pass HF_TOKEN at build time to avoid HuggingFace unauthenticated rate limits:
#   docker buildx build --build-arg HF_TOKEN=$HF_TOKEN ...
# A mirror copy is also stored in
# s3://chunking-models-902451183446-apsouth1/huggingface/ (ap-south-1) in case
# we need to rebuild without HuggingFace (e.g. if HF goes down).
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" \
 && docling-tools models download layout tableformer \
 && rm -rf /root/.cache/huggingface/xet \
 && find /root/.cache/huggingface -name "*.log" -delete

COPY src/ ./src/
COPY run_pipeline.py .

# Data + logs written at runtime.
RUN mkdir -p data/raw data/interim data/chunks data/embeddings \
             outputs/figures outputs/metrics logs

ENV DASH_PORT=17842
# API_KEY left unset at build time. Set it at deploy:
#   docker run -e API_KEY=... -p 17842:17842 sop-chunker
ENV PYTHONUNBUFFERED=1

EXPOSE 17842

CMD ["python", "-m", "src.ui.server"]
