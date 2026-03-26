# CLAUDE.md — Ollama Smart Router

## What this project is

A web app that routes user questions to the best local Ollama model.
A fast classifier reads the question, picks a model, and the response
streams to the browser in real time with routing reasoning visible.

This is a learning project. I'm a beginner developer building this to
understand how FastAPI, Ollama's API, and streaming work. Explain key
decisions and new concepts when you introduce them — not every line,
but the "why" behind architectural choices.

## Architecture (locked in — don't change these)

- **Backend:** FastAPI + Uvicorn, single file (main.py) unless it exceeds ~500 lines
- **Frontend:** Plain HTML + vanilla JavaScript in index.html (split from main.py at ~600 lines)
- **Streaming:** Server-Sent Events (SSE) via sse-starlette (not WebSockets)
- **HTTP client:** httpx (for calling Ollama)
- **Dependency management:** uv (never pip)

## Models

- **Classifier:** `llama3.2:3b` — tiny (~2GB VRAM), stays loaded alongside response models
- **General route:** `qwen3.5:35b-a3b` — MoE, fast, general knowledge
- **Code route:** `qwen3.5:27b` — dense, strong at programming and analysis
- **Reasoning route:** `deepseek-r1:32b` — chain-of-thought, math, logic
- **Ollama API:** `http://localhost:11434`

## Routing logic

The classifier sends the question to llama3.2:3b with a system prompt
that returns JSON: `{"route": "general|code|reasoning", "reason": "..."}`.
Classification uses stream=false, temperature=0, think=false, num_ctx=1024.
The actual response uses stream=true with num_ctx=4096.

## Rules

- Keep it simple. Fewer files, fewer abstractions, fewer dependencies.
- Don't add dependencies I haven't asked for.
- Don't refactor working code unless I ask you to.
- When something is new to me (SSE, async generators, Ollama's chunk format),
  add a short comment block explaining it.
- If you think my approach is wrong, say so and explain why — but don't
  silently change the plan.

## Tech conventions

- Python 3.13 (managed by uv)
- Type hints on all functions
- Ruff for formatting and linting
- f-strings, not .format()
- async functions for all endpoints and Ollama calls

## Build sequence

This project follows a six-step build plan (see ollama-router-project.md).
Each step has a specific scope. Don't build ahead — only implement what
the current step asks for.
