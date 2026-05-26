# VLM Chatbot

Two-container stack — FastAPI serves [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), Gradio renders the chat UI. Stateless API, client-side history. Cross-platform (Linux / macOS / Windows) via Docker Compose.

## Quick start (Docker)

```bash
docker compose up --build
```

Then open <http://localhost:7860>.

First boot downloads the model (~500 MB) into a named volume `huggingface-cache`, so subsequent runs start in seconds.

To stop and drop the model cache:

```bash
docker compose down -v
```

## Local development (with `uv`)

Two independent components, each with its own `pyproject.toml` + `uv.lock`.

**API** (port 8000):

```bash
cd api
uv sync
uv run uvicorn main:app --reload --port 8000
```

**UI** (port 7860, talks to the API on `localhost:8000` by default):

```bash
cd ui
uv sync
uv run python app.py
```

Override the API location with `API_URL`:

```bash
API_URL=http://localhost:8000 uv run python app.py
```

## Endpoints

- `GET /health` → `{"status": "ok"}`
- `POST /chat` — request schema in [api/schemas.py](api/schemas.py).

Smoke test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

## Cross-platform notes

- **Linux**: if you're not in the `docker` group, prefix commands with `sudo`. No `host.docker.internal` workaround needed — the UI reaches the API by service name `api`.
- **macOS / Windows**: requires Docker Desktop. Apple Silicon supported (the `python:3.11-slim` and `uv` base images are multi-arch).
- **Ports**: 7860 (UI) and 8000 (API) must be free.

## Layout

```
vlm_chatbot/
├── api/        # FastAPI + SmolVLM
├── ui/         # Gradio
└── docker-compose.yml
```
