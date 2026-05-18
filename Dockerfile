FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME="/app/.cache/huggingface" \
    SENTENCE_TRANSFORMERS_HOME="/app/.cache/sentence_transformers"

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

# Pre-download BGE-M3 model to bake into the image
RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Copy app source
COPY . .

EXPOSE 8000

# Default to development with auto-reload; override with compose for production
CMD ["uv", "run", "uvicorn", "app.api.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]