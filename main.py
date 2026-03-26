# FastAPI is a web framework — it lets you define URL endpoints (like GET /)
# and handles turning Python objects into JSON responses, HTML pages, etc.
#
# Uvicorn is an ASGI server — it's the thing that actually listens on a port
# (like localhost:8000), accepts incoming HTTP requests from browsers, and
# hands them to FastAPI to process. FastAPI defines *what* to do; Uvicorn
# handles the *networking*.

import json
import time
from collections.abc import AsyncGenerator

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

app = FastAPI()

OLLAMA_BASE_URL = "http://localhost:11434"

# --- Model routing map ---
#
# The classifier picks a route ("general", "code", or "reasoning"), and we
# map that to a specific Ollama model. The classifier itself always runs on
# the fast MoE model (qwen3.5:35b-a3b). When the route is "general", the
# response model happens to be the same as the classifier — that's fine,
# they're separate calls with different prompts (classification vs. answering).
CLASSIFIER_MODEL = "qwen3.5:35b-a3b"

ROUTE_TO_MODEL: dict[str, str] = {
    "general": "qwen3.5:35b-a3b",
    "code": "qwen3.5:27b",
    "reasoning": "deepseek-r1:32b",
}

CLASSIFIER_SYSTEM_PROMPT = """You are a question classifier. Given a user's question, decide which \
model should answer it. Respond with ONLY a JSON object, nothing else.

Your options:
- "general" — everyday questions, summaries, creative writing, quick facts
- "code" — programming, debugging, technical explanations, system design
- "reasoning" — math, logic puzzles, step-by-step analysis, comparisons

Format: {"route": "general|code|reasoning", "reason": "one sentence why"}

Examples:
- "What's the capital of France?" → {"route": "general", "reason": "Simple factual question"}
- "Write a Python function to sort a list" → {"route": "code", "reason": "Programming task"}
- "If I have 3 boxes with 2 balls each..." → {"route": "reasoning", "reason": "Logic problem requiring step-by-step thinking"}"""


class ChatRequest(BaseModel):
    message: str


# --- How Server-Sent Events (SSE) work ---
#
# SSE is a one-way streaming protocol: the server pushes data to the browser
# over a long-lived HTTP connection. It's simpler than WebSockets because:
#   - It uses plain HTTP (no upgrade handshake, no special protocol)
#   - The browser has a built-in API for it (EventSource / fetch ReadableStream)
#   - It's just text lines — easy to debug with curl or browser DevTools
#
# On the wire, each event looks like this:
#
#   data: Hello            ← one event carrying the text "Hello"
#   \n                     ← blank line = end of this event
#   data: world            ← next event
#   \n
#   data: [DONE]           ← our custom signal that the stream is finished
#   \n
#
# The "data:" prefix is part of the SSE spec. The browser's EventSource API
# automatically parses these lines and fires an onmessage callback for each.
# We use fetch + ReadableStream instead of EventSource because EventSource
# only supports GET requests and we need POST (to send the question body).
#
# --- How Ollama's streaming chunks work ---
#
# When you call Ollama's /api/chat with "stream": true, instead of one big
# JSON response, it sends many small JSON objects, one per line (NDJSON):
#
#   {"model":"qwen3.5:35b-a3b","message":{"role":"assistant","content":"Py"},"done":false}
#   {"model":"qwen3.5:35b-a3b","message":{"role":"assistant","content":"thon"},"done":false}
#   ...
#   {"model":"qwen3.5:35b-a3b","message":{"role":"assistant","content":""},"done":true}
#
# Each chunk has "done": false and carries a small piece of text in
# message.content. The final chunk has "done": true and empty content.
# We read each chunk, extract the token text, and forward it as an SSE event.
#
# --- Ollama's timing metadata (in the final chunk) ---
#
# When done=true, Ollama includes performance stats:
#
#   eval_count:    number of tokens the model generated (output tokens)
#   eval_duration: time spent generating those tokens, in *nanoseconds*
#
# Tokens-per-second = eval_count / (eval_duration / 1e9)
#
# This measures pure generation speed — excludes prompt processing,
# model loading, and network overhead. It's the best measure of how
# fast the model itself is running on your hardware.


# --- Why classification uses stream=false, temperature=0, and think=false ---
#
# stream=false: Classification is a quick, small response (just a JSON object).
# We need the full response before we can pick a model and start streaming,
# so there's no benefit to streaming it — we'd just wait for all chunks anyway.
#
# temperature=0: We want deterministic routing. The same question should always
# go to the same model. Temperature=0 removes randomness from sampling, so the
# classifier picks the most likely route every time instead of occasionally
# rolling a different answer.
#
# think=false: Some models (like qwen3.5) have a "thinking" mode where they
# generate hundreds of internal reasoning tokens before the actual answer.
# That's great for complex questions but terrible for classification — it
# turns a sub-second call into 15+ seconds of unnecessary chain-of-thought.
# Disabling it forces the model to respond directly with the JSON we need.


async def classify_question(question: str) -> tuple[str, str, str, float]:
    """Classify a question and return (route, reason, model_name, duration_seconds).

    Sends the question to the classifier model with a system prompt that
    asks for a JSON response. If parsing fails, defaults to "general".
    """
    start = time.monotonic()

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CLASSIFIER_MODEL,
                "messages": [
                    {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                "stream": False,
                "think": False,
                "options": {"temperature": 0},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

    classify_time = time.monotonic() - start
    content = data["message"]["content"]

    # The classifier should return pure JSON, but sometimes models wrap it
    # in markdown code fences or add extra text. Try to extract JSON.
    try:
        result = json.loads(content)
        route = result["route"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        # If the model returned something unparseable, fall back to general.
        # This keeps the app working even if the classifier misbehaves.
        route = "general"
        reason = "Classification unclear, using default"

    # Validate that the route is one we know about
    if route not in ROUTE_TO_MODEL:
        route = "general"
        reason = "Classification unclear, using default"

    model = ROUTE_TO_MODEL[route]
    return route, reason, model, classify_time


async def stream_chat(message: str) -> AsyncGenerator[dict[str, str], None]:
    """Classify the question, then stream the response from the chosen model.

    SSE event types sent to the frontend:
      1. "classification" — route info + classify time (immediately)
      2. default (data-only) — token text chunks
      3. "metrics" — timing summary (after final token)
      4. data: [DONE] — signals end of stream
    """
    # Step 1: Classify (non-streaming, fast)
    route, reason, model, classify_time = await classify_question(message)

    # Step 2: Check if the target model is already loaded in memory.
    # If not, Ollama will need to swap models, which can take minutes for
    # large models. We tell the frontend so it can show a loading message.
    model_loaded = True
    try:
        async with httpx.AsyncClient() as client:
            ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=5.0)
            ps_resp.raise_for_status()
            loaded = [m["name"] for m in ps_resp.json().get("models", [])]
            model_loaded = model in loaded
    except httpx.HTTPError:
        pass  # If we can't check, assume it's loaded and don't show the warning

    # Step 3: Send classification as the first SSE event
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

    # Step 4: Stream the actual response from the chosen model
    stream_start = time.monotonic()
    first_token_time: float | None = None

    # --- httpx timeout strategy for model swaps ---
    #
    # When Ollama switches models, it must unload one from VRAM and load
    # another from disk. For 20-30GB models this can take minutes — well
    # beyond a simple 120s total timeout.
    #
    # httpx.Timeout lets us set separate limits:
    #   connect: max time to establish the TCP connection (10s is plenty)
    #   read:    max time to wait *between* chunks (120s handles slow model loads,
    #            then resets after each streaming chunk arrives)
    #   write:   max time to send the request body (10s is plenty)
    #   pool:    max time waiting for a connection from the pool (10s)
    #
    # This way a model swap that takes 3 minutes won't timeout, but a truly
    # dead connection (no data for 120s) still gets caught.
    stream_timeout = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": True,
            },
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

                    # Extract Ollama's generation stats from the final chunk.
                    # eval_count = output tokens generated
                    # eval_duration = generation time in nanoseconds
                    eval_count = chunk.get("eval_count", 0)
                    eval_duration_ns = chunk.get("eval_duration", 0)

                    tokens_per_sec = 0.0
                    if eval_duration_ns > 0:
                        tokens_per_sec = eval_count / (eval_duration_ns / 1e9)

                    # Step 4: Send timing metrics as a named event
                    yield {
                        "event": "metrics",
                        "data": json.dumps(
                            {
                                "classify_seconds": round(classify_time, 2),
                                "first_token_seconds": round(first_token_time or 0, 2),
                                "total_seconds": round(total_time, 2),
                                "tokens_per_second": round(tokens_per_sec, 1),
                                "eval_count": eval_count,
                            }
                        ),
                    }

                    yield {"data": "[DONE]"}
                    return


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(stream_chat(request.message))


@app.get("/models")
async def models() -> dict:
    """List available models and which ones are currently loaded in VRAM.

    Calls two Ollama endpoints:
      /api/tags — all models downloaded on disk
      /api/ps   — models currently loaded in memory (ready for fast inference)
    """
    async with httpx.AsyncClient() as client:
        tags_resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10.0)
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

        ps_resp = await client.get(f"{OLLAMA_BASE_URL}/api/ps", timeout=10.0)
        ps_resp.raise_for_status()
        loaded_models = [m["name"] for m in ps_resp.json().get("models", [])]

    return {"available": all_models, "loaded": loaded_models}


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ollama Smart Router</title>
        <style>
            body {
                font-family: system-ui, sans-serif;
                max-width: 700px;
                margin: 40px auto;
                padding: 0 20px;
            }
            #input-row {
                display: flex;
                gap: 8px;
            }
            #question {
                flex: 1;
                padding: 10px;
                font-size: 16px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }
            #send-btn {
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 6px;
                background: #2563eb;
                color: white;
                cursor: pointer;
            }
            #send-btn:disabled {
                background: #93b4f5;
                cursor: not-allowed;
            }
            #status {
                margin-top: 12px;
                color: #666;
                font-style: italic;
            }
            #routing-badge {
                margin-top: 12px;
                padding: 8px 12px;
                background: #e0f2fe;
                border-left: 3px solid #2563eb;
                border-radius: 4px;
                font-size: 14px;
                color: #1e40af;
                display: none;
            }
            #routing-badge .model-name {
                font-weight: 600;
            }
            #routing-badge .route-reason {
                color: #64748b;
                margin-left: 4px;
            }
            #response {
                margin-top: 16px;
                padding: 16px;
                background: #f4f4f5;
                border-radius: 8px;
                white-space: pre-wrap;
                display: none;
            }
            #timing {
                margin-top: 8px;
                font-size: 13px;
                color: #94a3b8;
                display: none;
            }
            #loaded-models {
                margin-top: 24px;
                padding: 8px 12px;
                background: #f8fafc;
                border: 1px solid #e2e8f0;
                border-radius: 6px;
                font-size: 13px;
                color: #64748b;
            }
            #loaded-models .label {
                font-weight: 600;
                color: #475569;
            }
            #loaded-models .model-tag {
                display: inline-block;
                padding: 2px 8px;
                margin: 2px 4px;
                background: #e0f2fe;
                border-radius: 4px;
                font-size: 12px;
                color: #1e40af;
            }
            #loaded-models .none-loaded {
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <h1>Ollama Smart Router</h1>

        <div id="input-row">
            <input type="text" id="question" placeholder="Ask a question..." />
            <button id="send-btn" onclick="sendQuestion()">Send</button>
        </div>
        <div id="status"></div>
        <div id="routing-badge"></div>
        <div id="response"></div>
        <div id="timing"></div>
        <div id="loaded-models"><span class="label">Models in memory:</span> loading...</div>

        <script>
            const questionInput = document.getElementById("question");
            const sendBtn = document.getElementById("send-btn");
            const statusDiv = document.getElementById("status");
            const routingBadge = document.getElementById("routing-badge");
            const responseDiv = document.getElementById("response");
            const timingDiv = document.getElementById("timing");
            const loadedModelsDiv = document.getElementById("loaded-models");

            // Send on Enter key press
            questionInput.addEventListener("keydown", (e) => {
                if (e.key === "Enter") sendQuestion();
            });

            // --- Streaming with fetch + ReadableStream ---
            //
            // We can't use the browser's built-in EventSource API here because
            // EventSource only supports GET requests. We need POST to send the
            // question in the request body. Instead, we use fetch() and manually
            // read the response body as a stream of text.
            //
            // The SSE stream now has these event types:
            //   1. "classification" — JSON with route, reason, model, classify time
            //   2. default events   — token text chunks
            //   3. "metrics"        — timing summary (after last token)
            //   4. data: [DONE]     — signals end of stream

            async function sendQuestion() {
                const message = questionInput.value.trim();
                if (!message) return;

                sendBtn.disabled = true;
                statusDiv.textContent = "Classifying question...";
                routingBadge.style.display = "none";
                responseDiv.textContent = "";
                responseDiv.style.display = "none";
                timingDiv.style.display = "none";

                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({message}),
                    });

                    if (!res.ok) throw new Error("Request failed");

                    const reader = res.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = "";

                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, {stream: true});

                        // Split on blank lines — handles both \\r\\n and \\n
                        const parts = buffer.split(/\\r?\\n\\r?\\n/);
                        buffer = parts.pop();

                        for (const part of parts) {
                            let eventType = "message";
                            let eventData = "";

                            for (const line of part.split(/\\r?\\n/)) {
                                if (line.startsWith("event: ")) {
                                    eventType = line.slice(7);
                                } else if (line.startsWith("data: ")) {
                                    eventData = line.slice(6);
                                }
                            }

                            // Classification event — show routing badge
                            if (eventType === "classification") {
                                const info = JSON.parse(eventData);
                                routingBadge.innerHTML =
                                    'Routed to: <span class="model-name">' + info.model + '</span>' +
                                    '<span class="route-reason"> (' + info.reason + ')</span>';
                                routingBadge.style.display = "block";
                                if (info.model_loaded) {
                                    statusDiv.textContent = "Streaming response...";
                                } else {
                                    statusDiv.textContent = "Loading " + info.model + " into memory (this may take a minute)...";
                                }
                                continue;
                            }

                            // Metrics event — show timing info
                            if (eventType === "metrics") {
                                const m = JSON.parse(eventData);
                                timingDiv.textContent =
                                    "Classified in " + m.classify_seconds + "s" +
                                    " | First token: " + m.first_token_seconds + "s" +
                                    " | Total: " + m.total_seconds + "s" +
                                    " | " + m.tokens_per_second + " tok/s" +
                                    " (" + m.eval_count + " tokens)";
                                timingDiv.style.display = "block";
                                continue;
                            }

                            // [DONE] — finish up and refresh loaded models
                            if (eventData === "[DONE]") {
                                statusDiv.textContent = "";
                                sendBtn.disabled = false;
                                questionInput.focus();
                                refreshModels();
                                return;
                            }

                            // Token event — append to response
                            if (eventData) {
                                if (responseDiv.style.display === "none") {
                                    statusDiv.textContent = "";
                                    responseDiv.style.display = "block";
                                }
                                responseDiv.textContent += eventData;
                            }
                        }
                    }
                } catch (err) {
                    statusDiv.textContent = "Error: " + err.message;
                }

                sendBtn.disabled = false;
                questionInput.focus();
            }

            // Fetch which models are currently loaded in Ollama's memory
            async function refreshModels() {
                try {
                    const res = await fetch("/models");
                    if (!res.ok) return;
                    const data = await res.json();

                    let html = '<span class="label">Models in memory:</span> ';
                    if (data.loaded.length === 0) {
                        html += '<span class="none-loaded">none</span>';
                    } else {
                        html += data.loaded.map(
                            name => '<span class="model-tag">' + name + '</span>'
                        ).join("");
                    }
                    loadedModelsDiv.innerHTML = html;
                } catch (e) {
                    loadedModelsDiv.innerHTML =
                        '<span class="label">Models in memory:</span> <span class="none-loaded">could not reach Ollama</span>';
                }
            }

            // Load model status on page load
            refreshModels();

            // Auto-focus the input on page load
            questionInput.focus();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
