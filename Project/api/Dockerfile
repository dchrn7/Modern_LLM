FROM python:3.11-slim

# Pull the static uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /uvx /bin/

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

# Install deps first (cache-friendly), then copy code
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --frozen --no-install-project

#Le code nécessaire
COPY api/    /app/api/
COPY models/ /app/models/
COPY data/   /app/data/

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
