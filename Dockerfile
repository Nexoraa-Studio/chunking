FROM python:3.11-slim-bookworm

# libgomp1 is needed by scikit-learn / faiss. We no longer need libgl / libx*
# because docling (with its opencv stack) was removed.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps requirements.txt from pulling CUDA wheels.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.2,<3"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download MiniLM + nltk punkt so first PDF upload doesn't stall on a HF
# download. Pass HF_TOKEN at build time to avoid unauthenticated rate limits:
#   docker buildx build --build-arg HF_TOKEN=$HF_TOKEN ...
# Mirror copy is at
#   s3://chunking-models-902451183446-apsouth1/huggingface/ (ap-south-1).
# Docling's layout/table ML models are NOT baked — we use pymupdf as the
# default extractor, which needs zero ML models to work.
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')" \
 && python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)" \
 && rm -rf /root/.cache/huggingface/xet \
 && find /root/.cache/huggingface -name "*.log" -delete

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY run_pipeline.py .

# Data + logs written at runtime.
RUN mkdir -p data/raw data/interim data/chunks data/embeddings \
             outputs/figures outputs/metrics logs

ENV DASH_PORT=17842
# EXTRACTOR=pymupdf is the default; set EXTRACTOR=docling only if you also
# `pip install docling` into the container.
# API_KEY left unset at build time. Set at deploy:
#   docker run -e API_KEY=... -p 17842:17842 sop-chunker
ENV PYTHONUNBUFFERED=1

EXPOSE 17842

CMD ["python", "-m", "src.ui.server"]
