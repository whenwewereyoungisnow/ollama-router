# FastAPI is a web framework — it lets you define URL endpoints (like GET /)
# and handles turning Python objects into JSON responses, HTML pages, etc.
#
# Uvicorn is an ASGI server — it's the thing that actually listens on a port
# (like localhost:8000), accepts incoming HTTP requests from browsers, and
# hands them to FastAPI to process. FastAPI defines *what* to do; Uvicorn
# handles the *networking*.

import json
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


# --- Why classification uses stream=false and temperature=0 ---
#
# stream=false: Classification is a quick, small response (just a JSON object).
# We need the full response before we can pick a model and start streaming,
# so there's no benefit to streaming it — we'd just wait for all chunks anyway.
#
# temperature=0: We want deterministic routing. The same question should always
# go to the same model. Temperature=0 removes randomness from sampling, so the
# classifier picks the most likely route every time instead of occasionally
# rolling a different answer.


async def classify_question(question: str) -> tuple[str, str, str]:
    """Classify a question and return (route, reason, model_name).

    Sends the question to the classifier model with a system prompt that
    asks for a JSON response. If parsing fails, defaults to "general".
    """
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
                "options": {"temperature": 0},
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

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
    return route, reason, model


async def stream_chat(message: str) -> AsyncGenerator[dict[str, str], None]:
    """Classify the question, then stream the response from the chosen model.

    The first SSE event carries the classification result as JSON (with an
    "event" field set to "classification" so the frontend can distinguish it
    from token events). All subsequent events are token chunks.
    """
    # Step 1: Classify (non-streaming, fast)
    route, reason, model = await classify_question(message)

    # Step 2: Send classification as the first SSE event.
    # We use a named event type ("classification") so the frontend can tell
    # this apart from token data without inspecting the payload.
    yield {
        "event": "classification",
        "data": json.dumps({"route": route, "reason": reason, "model": model}),
    }

    # Step 3: Stream the actual response from the chosen model
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": message}],
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                chunk = json.loads(line)
                token = chunk["message"]["content"]

                if token:
                    yield {"data": token}

                if chunk.get("done"):
                    yield {"data": "[DONE]"}
                    return


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    return EventSourceResponse(stream_chat(request.message))


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

        <script>
            const questionInput = document.getElementById("question");
            const sendBtn = document.getElementById("send-btn");
            const statusDiv = document.getElementById("status");
            const routingBadge = document.getElementById("routing-badge");
            const responseDiv = document.getElementById("response");

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
            // The SSE stream now has two event types:
            //   1. "classification" event — JSON with route, reason, model
            //   2. default "message" events — token text chunks + [DONE]
            //
            // Named events look like this on the wire:
            //   event: classification
            //   data: {"route":"code","reason":"...","model":"qwen3.5:27b"}
            //
            // Default events (no "event:" line) just have:
            //   data: Hello

            async function sendQuestion() {
                const message = questionInput.value.trim();
                if (!message) return;

                sendBtn.disabled = true;
                statusDiv.textContent = "Classifying question...";
                routingBadge.style.display = "none";
                responseDiv.textContent = "";
                responseDiv.style.display = "none";

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
                    let currentModel = "";

                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;

                        buffer += decoder.decode(value, {stream: true});

                        // Split on blank lines — handles both \\r\\n and \\n
                        const parts = buffer.split(/\\r?\\n\\r?\\n/);
                        buffer = parts.pop();

                        for (const part of parts) {
                            // Parse each SSE event: look for "event:" and "data:" lines
                            let eventType = "message";
                            let eventData = "";

                            for (const line of part.split(/\\r?\\n/)) {
                                if (line.startsWith("event: ")) {
                                    eventType = line.slice(7);
                                } else if (line.startsWith("data: ")) {
                                    eventData = line.slice(6);
                                }
                            }

                            // Handle classification event — show the routing badge
                            if (eventType === "classification") {
                                const info = JSON.parse(eventData);
                                currentModel = info.model;
                                routingBadge.innerHTML =
                                    'Routed to: <span class="model-name">' + info.model + '</span>' +
                                    '<span class="route-reason"> (' + info.reason + ')</span>';
                                routingBadge.style.display = "block";
                                statusDiv.textContent = "Streaming response...";
                                continue;
                            }

                            // Handle token events
                            if (eventData === "[DONE]") {
                                statusDiv.textContent = "";
                                sendBtn.disabled = false;
                                questionInput.focus();
                                return;
                            }

                            if (eventData) {
                                // First token: show the response area
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
