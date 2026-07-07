# Production image for the LocalMind backend.
# Build context is this backend/ directory (deploy with `--source backend`), so
# the frontend is never copied into the image.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# LLM_PROVIDER + the provider's API key are injected at deploy time (secrets).
# Defaults to ollama, but a container has no Ollama — set LLM_PROVIDER=openrouter
# (or groq/azure) in the environment.
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
