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
CHAT_MODEL = "qwen3.5:35b-a3b"


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


async def stream_chat(message: str) -> AsyncGenerator[dict[str, str], None]:
    """Stream tokens from Ollama and yield them as SSE event dicts.

    Each yielded dict has a "data" key — sse-starlette turns these into
    properly formatted SSE lines (data: ...\n\n) automatically.
    """
    async with httpx.AsyncClient() as client:
        # stream=True tells httpx to give us the response incrementally
        # (not to be confused with the stream=True in the JSON body, which
        # tells *Ollama* to stream its output). We need both.
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": message}],
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()

            # Ollama sends newline-delimited JSON (one JSON object per line).
            # httpx's aiter_lines() gives us each line as it arrives.
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
        <div id="response"></div>

        <script>
            const questionInput = document.getElementById("question");
            const sendBtn = document.getElementById("send-btn");
            const statusDiv = document.getElementById("status");
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
            // The response is an SSE stream — lines like "data: Hello\\n\\n".
            // We split on double-newlines to get individual events, strip the
            // "data: " prefix, and append each token to the page.

            async function sendQuestion() {
                const message = questionInput.value.trim();
                if (!message) return;

                sendBtn.disabled = true;
                statusDiv.textContent = "Thinking...";
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

                    while (true) {
                        const {done, value} = await reader.read();
                        if (done) break;

                        // Decode the binary chunk into text and add to buffer.
                        // SSE events are separated by blank lines. We use a
                        // regex to handle both \\r\\n and \\n line endings
                        // (sse-starlette sends \\r\\n).
                        buffer += decoder.decode(value, {stream: true});

                        // Split on blank lines (two consecutive line breaks)
                        const parts = buffer.split(/\\r?\\n\\r?\\n/);
                        // Keep the last part — it might be an incomplete event
                        buffer = parts.pop();

                        for (const part of parts) {
                            for (const line of part.split(/\\r?\\n/)) {
                                if (!line.startsWith("data: ")) continue;
                                const token = line.slice(6);  // strip "data: "

                                if (token === "[DONE]") {
                                    statusDiv.textContent = "Model: __MODEL__";
                                    sendBtn.disabled = false;
                                    questionInput.focus();
                                    return;
                                }

                                // First token: clear "Thinking..." and show response area
                                if (responseDiv.style.display === "none") {
                                    statusDiv.textContent = "";
                                    responseDiv.style.display = "block";
                                }

                                responseDiv.textContent += token;
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
    """.replace("__MODEL__", CHAT_MODEL)
    return HTMLResponse(content=html)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
