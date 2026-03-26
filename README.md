# Ollama Smart Router

A local AI chat app that automatically routes your questions to the best model. Ask anything — a lightweight classifier reads your question, picks the right model, and streams the response to your browser in real time.

## How it works

```
You type a question
        |
        v
  llama3.2:3b (classifier)
  Reads the question, picks a route
        |
        +---> general  --> qwen3.5:35b-a3b  (fast, general knowledge)
        +---> code     --> qwen3.5:27b       (programming, scripts)
        +---> reasoning--> deepseek-r1:32b   (math, logic, chain-of-thought)
        |
        v
  Response streams to your browser via SSE
```

The classifier is tiny (~2 GB VRAM) and stays loaded alongside the response model, so there's no model-swapping delay between questions.

## Features

- **Automatic routing** — no manual model selection needed
- **Real-time streaming** — tokens appear as they're generated (Server-Sent Events)
- **Routing transparency** — see which model was chosen and why
- **Performance metrics** — classification time, time-to-first-token, tokens/sec
- **Conversation history** — multi-turn chat with context
- **No external APIs** — everything runs locally via Ollama

## Prerequisites

- [Python 3.13+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) (Python package manager)
- [Ollama](https://ollama.com/) running locally

### Pull the required models

```bash
ollama pull llama3.2:3b
ollama pull qwen3.5:35b-a3b
ollama pull qwen3.5:27b
ollama pull deepseek-r1:32b
```

## Quickstart

```bash
# Clone the repo
git clone https://github.com/fabianferdinand/ollama-router.git
cd ollama-router

# Install dependencies
uv sync

# Start the server
uv run uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Tech stack

| Layer | Tool |
|-------|------|
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML + JS |
| Streaming | SSE (sse-starlette) |
| HTTP client | httpx |
| LLM runtime | Ollama |
| Package manager | uv |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serves the chat UI |
| `POST` | `/chat` | Streams a routed response (SSE) |
| `POST` | `/clear` | Clears conversation (client-side) |
| `GET` | `/models` | Lists available and loaded models |
| `GET` | `/health` | Health check |

## Project structure

```
ollama-router/
  main.py        # FastAPI backend — routing, classification, streaming
  index.html     # Chat UI — vanilla HTML/JS
  pyproject.toml # Dependencies and project metadata
```

Intentionally simple: two files, zero abstractions. This is a learning project.

## License

MIT
