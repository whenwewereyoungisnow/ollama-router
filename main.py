# --- Ollama Smart Router — Backend ---
#
# Splitting into main.py + index.html because the combined file exceeded
# 600 lines. The Python backend handles routing logic, Ollama communication,
# and SSE streaming. The HTML file handles all UI concerns. Neither has any
# knowledge of the other's internals — they communicate via JSON over HTTP.

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, field_validator
from sse_starlette.sse import EventSourceResponse

# Configure the classifier debug logger. Uvicorn's default log config
# suppresses other loggers, so we need to set up our own handler.
logger = logging.getLogger("router")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(_handler)

OLLAMA_BASE_URL = "http://localhost:11434"


# --- Shared HTTP client (connection pooling) ---
#
# Without this, every call to check_ollama(), classify_question(), and
# stream_chat() creates a brand-new httpx.AsyncClient — which means a
# fresh TCP connection each time. That adds ~5-20ms of connection setup
# per request. A shared client keeps connections alive and reuses them,
# which matters when we make 2-3 Ollama calls per user message.
#
# FastAPI's "lifespan" is the standard way to manage resources that live
# for the entire app lifetime: create on startup, clean up on shutdown.
http_client: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(base_url=OLLAMA_BASE_URL)
    yield
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)

# --- Why a tiny classifier model? ---
#
# Classification is a simple task: read a question, pick one of three labels,
# return one line of JSON. A 3B model handles this easily. The big advantage
# is memory: llama3.2:3b uses ~2GB of VRAM, small enough to stay loaded
# alongside whichever large response model is active. With the 35b classifier
# we had before, Ollama often had to unload it to make room for the response
# model, then reload it for the next classification — adding 10-30s of model
# swapping to every question. A dedicated small classifier avoids that.
CLASSIFIER_MODEL = "llama3.2:3b"

# --- Thinking mode ---
#
# Some models (qwen3.5, deepseek-r1) have a "thinking" mode where they
# generate a hidden chain-of-thought before the visible response. This is
# like a scratchpad — the model reasons internally, then gives a polished
# answer. It improves quality on hard problems but adds significant latency
# (often 10-30s of extra generation you never see in the output).
#
# We disable thinking for the general and code models because their questions
# don't need it — a factual answer or code snippet doesn't benefit from
# hidden reasoning, and the speed penalty isn't worth it. deepseek-r1 is
# the exception: its entire value is chain-of-thought reasoning, and its
# thinking *is* visible in the response (the <think>...</think> blocks).
# That's the whole point of routing reasoning questions to it.

ROUTE_TO_MODEL: dict[str, str] = {
    "general": "qwen3.5:35b-a3b",
    "code": "qwen3.5:27b",
    "reasoning": "deepseek-r1:32b",
}

# --- Classifier prompt design for small models ---
#
# Small models (3B) struggle with JSON output — they often add markdown
# fences, extra text, or malformed syntax. A plain-text format (route on
# line 1, reason on line 2) is much more reliable. We parse by just reading
# the first line and checking if it's a known route.
CLASSIFIER_SYSTEM_PROMPT = """You are a router. Do NOT answer the question. Just classify it.
Reply with EXACTLY two lines. Nothing else.
Line 1: general, code, or reasoning
Line 2: why

general = facts, advice, creative writing, explanations
code = writing code, debugging, programming, scripts, system design
reasoning = math, logic, puzzles, probability, word problems

Q: What color is the sky?
general
Factual question

Q: Write a Python function to sort a list
code
Programming task

Q: Write a bash script to rename files
code
Scripting task

Q: Explain how HTTP works
general
Technical explanation

Q: If I have 3 boxes with 2 balls, how many combinations?
reasoning
Combinatorics problem

Q: What's the probability of rolling two sixes?
reasoning
Probability calculation

Q: Write a Python function that checks if a word is a palindrome
code
Programming task"""


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[dict[str, str]]) -> list[dict[str, str]]:
        if not v:
            raise ValueError("messages must contain at least one message")
        return v


# --- Ollama timing metadata ---
# The final streaming chunk includes:
#   eval_count:    number of output tokens generated
#   eval_duration: generation time in nanoseconds
# Tokens/sec = eval_count / (eval_duration / 1e9)
# This measures pure generation speed, excluding prompt processing and model loading.

# --- num_ctx: controlling the context window ---
#
# num_ctx sets the maximum number of tokens (prompt + response) the model
# can work with in a single call. Ollama defaults to a large context window
# (often 4096-131072 depending on the model), but bigger context windows use
# more VRAM — the memory needed scales with context size because of the
# KV cache (the attention mechanism's working memory).
#
# By setting num_ctx explicitly we:
#   1. Use less VRAM → models load faster, less swapping between models
#   2. Get slightly faster inference → smaller KV cache = faster attention
#   3. Stay predictable → same context size regardless of model defaults
#
# Classifier: 1024 is plenty. The system prompt + question + JSON response
# is well under 500 tokens. No reason to allocate a huge context window.
#
# Chat response: 4096 gives room for multi-turn conversations. If you start
# hitting the limit (model forgets early messages), bump this up — but it
# costs VRAM and speed.


async def check_ollama() -> tuple[bool, list[str]]:
    """Check if Ollama is reachable and return (ok, list_of_loaded_models)."""
    try:
        ps_resp = await http_client.get("/api/ps", timeout=5.0)
        ps_resp.raise_for_status()
        loaded = [m["name"] for m in ps_resp.json().get("models", [])]
        return True, loaded
    except (httpx.ConnectError, httpx.HTTPError):
        return False, []


async def classify_question(question: str) -> tuple[str, str, str, float]:
    """Classify a question → (route, reason, model_name, duration_seconds).

    Uses stream=false (we need the full answer before picking a model),
    temperature=0 (deterministic routing), and think=false (skip the model's
    internal chain-of-thought which wastes 15+ seconds on a trivial JSON task).
    """
    start = time.monotonic()

    response = await http_client.post(
        "/api/chat",
        json={
            "model": CLASSIFIER_MODEL,
            # We wrap the question in "Classify: ..." instead of sending
            # it as a bare user message. Without this, small models see
            # "Write a Python function..." and answer it instead of
            # classifying it — the instruction-tuning is too strong.
            "messages": [
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify: {question}"},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": 1024},
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    classify_time = time.monotonic() - start
    content = data["message"]["content"].strip()

    # Debug log — shows in the uvicorn terminal so you can see exactly
    # what the classifier returned when things go wrong.
    logger.info("classify question=%r", question)
    logger.info("classify raw=%r", content)

    # Parse the plain-text response. The first line should contain the route
    # keyword, but small models often add extra text like "Line 1: general"
    # or "Category: code". We scan for any known route keyword in the first
    # line rather than expecting an exact match.
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    first_line = lines[0].lower() if lines else ""
    # Clean the reason line — strip prefixes like "Line 2:" that the model
    # sometimes echoes back from the prompt instructions.
    raw_reason = lines[1] if len(lines) > 1 else "No reason given"
    reason = (
        raw_reason.split(":", 1)[-1].strip()
        if ":" in raw_reason and len(raw_reason.split(":", 1)[0]) < 10
        else raw_reason
    )

    route = ""
    for candidate in ROUTE_TO_MODEL:
        if candidate in first_line:
            route = candidate
            break

    if not route:
        logger.warning(
            "classify no known route in first line=%r, falling back to general",
            first_line,
        )
        route = "general"
        reason = "Classification unclear, using default"

    logger.info("classify route=%s model=%s", route, ROUTE_TO_MODEL[route])
    return route, reason, ROUTE_TO_MODEL[route], classify_time


async def stream_chat(
    messages: list[dict[str, str]],
) -> AsyncGenerator[dict[str, str], None]:
    """Classify, then stream the response from the chosen model.

    SSE event types: "classification", default (tokens), "metrics", then [DONE].
    On any error, yields an "error" event so the frontend can display it.
    """
    # Check Ollama is reachable before doing anything
    ollama_ok, loaded_models = await check_ollama()
    if not ollama_ok:
        yield {
            "event": "error",
            "data": json.dumps(
                {"message": "Cannot connect to Ollama at " + OLLAMA_BASE_URL}
            ),
        }
        return

    # Classify based on the latest user message
    last_user_msg = messages[-1]["content"]

    try:
        route, reason, model, classify_time = await classify_question(last_user_msg)
    except httpx.ConnectError:
        yield {
            "event": "error",
            "data": json.dumps(
                {"message": "Lost connection to Ollama during classification"}
            ),
        }
        return
    except Exception as e:
        # Classification failed but we can still answer with the default model
        logger.warning("classify error: %s", e)
        route, reason = "general", "Classification failed, using default"
        model = ROUTE_TO_MODEL["general"]
        classify_time = 0.0

    model_loaded = model in loaded_models

    yield {
        "event": "classification",
        "data": json.dumps(
            {
                "route": route,
                "reason": reason,
                "model": model,
                "classify_seconds": round(classify_time, 2),
                "model_loaded": model_loaded,
            }
        ),
    }

    # Stream the response. Separate timeouts so model swaps don't get killed —
    # the read timeout resets after each chunk, so only a truly dead connection
    # (no data for 120s) triggers a timeout.
    stream_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    stream_start = time.monotonic()
    first_token_time: float | None = None

    try:
        # Disable thinking for non-reasoning routes (see comment above)
        chat_payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"num_ctx": 4096},
        }
        if route != "reasoning":
            chat_payload["think"] = False

        async with http_client.stream(
            "POST",
            "/api/chat",
            json=chat_payload,
            timeout=stream_timeout,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk["message"]["content"]

                if token:
                    if first_token_time is None:
                        first_token_time = time.monotonic() - stream_start
                    yield {"data": token}

                if chunk.get("done"):
                    total_time = time.monotonic() - stream_start
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)
                    tps = (
                        eval_count / (eval_duration_ns / 1e9)
                        if eval_duration_ns > 0
                        else 0.0
                    )

                    yield {
                        "event": "metrics",
                        "data": json.dumps(
                            {
                                "classify_seconds": round(classify_time, 2),
                                "first_token_seconds": round(first_token_time or 0, 2),
                                "total_seconds": round(total_time, 2),
                                "tokens_per_second": round(tps, 1),
                                "eval_count": eval_count,
                            }
                        ),
                    }
                    yield {"data": "[DONE]"}
                    return

    except httpx.ConnectError:
        yield {
            "event": "error",
            "data": json.dumps(
                {"message": "Lost connection to Ollama during streaming"}
            ),
        }
    except httpx.HTTPStatusError as e:
        # Ollama returns 404 when a model isn't downloaded
        msg = f"Model '{model}' not found — run: ollama pull {model}"
        if e.response.status_code != 404:
            msg = f"Ollama error {e.response.status_code}: {e.response.text[:200]}"
        yield {"event": "error", "data": json.dumps({"message": msg})}


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(stream_chat(request.messages))


@app.post("/clear")
async def clear() -> dict[str, str]:
    """No server state to clear — conversation lives in the browser.
    This endpoint exists so the frontend has a clean semantic action."""
    return {"status": "cleared"}


@app.get("/models")
async def models() -> dict:
    """List available and loaded models from Ollama."""
    try:
        tags_resp = await http_client.get("/api/tags", timeout=10.0)
        tags_resp.raise_for_status()
        all_models = [
            {
                "name": m["name"],
                "size_gb": round(m["size"] / 1e9, 1),
                "parameter_size": m["details"].get("parameter_size", ""),
                "quantization": m["details"].get("quantization_level", ""),
            }
            for m in tags_resp.json().get("models", [])
        ]

        ps_resp = await http_client.get("/api/ps", timeout=10.0)
        ps_resp.raise_for_status()
        loaded = [m["name"] for m in ps_resp.json().get("models", [])]

        return {"available": all_models, "loaded": loaded}
    except (httpx.ConnectError, httpx.HTTPError):
        return {"available": [], "loaded": [], "error": "Cannot reach Ollama"}


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
