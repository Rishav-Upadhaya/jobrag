FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

# Copy dependency metadata
COPY pyproject.toml ./
COPY uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Suppress HF hub progress bars and ensure transformers can run during build
ENV TRANSFORMERS_OFFLINE=0
ENV HF_HUB_DISABLE_PROGRESS_BARS=1

# Pre-download SentenceTransformer model into the image layer to avoid
# runtime downloads on cold start.
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

# Prevent the model from making HTTP calls to HuggingFace on every query
# to check for updates — this cuts retriever time by 4-6 seconds alone
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_OFFLINE=1

# Copy app source
COPY . .

EXPOSE 8000

# Default to development with auto-reload; override with compose for production
CMD ["uv", "run", "uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]