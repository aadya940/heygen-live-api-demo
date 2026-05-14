FROM python:3.11-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Cache the TF Hub model at build time so cold starts are instant
ENV TFHUB_CACHE_DIR=/app/.tfhub_cache
RUN python -c "\
import tensorflow_hub as hub; \
hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')"

COPY movenet.py backend.py ./

EXPOSE 8000

CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
