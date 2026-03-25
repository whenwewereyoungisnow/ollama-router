# FastAPI is a web framework — it lets you define URL endpoints (like GET /)
# and handles turning Python objects into JSON responses, HTML pages, etc.
#
# Uvicorn is an ASGI server — it's the thing that actually listens on a port
# (like localhost:8000), accepts incoming HTTP requests from browsers, and
# hands them to FastAPI to process. FastAPI defines *what* to do; Uvicorn
# handles the *networking*.

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI()

OLLAMA_BASE_URL = "http://localhost:11434"
CHAT_MODEL = "qwen3.5:35b-a3b"


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    model: str


# --- How Ollama's /api/chat endpoint works ---
#
# REQUEST: You POST JSON with these fields:
#   {
#     "model": "qwen3.5:35b-a3b",   — which model to use
#     "messages": [                   — conversation history in OpenAI-style format
#       {"role": "user", "content": "What is Python?"}
#     ],
#     "stream": false                 — if true, Ollama sends chunks as they're generated
#                                       (like ChatGPT's typing effect). We use false here
#                                       to get the full response in one shot.
#   }
#
# RESPONSE (when stream=false): A single JSON object:
#   {
#     "model": "qwen3.5:35b-a3b",
#     "message": {"role": "assistant", "content": "Python is a ..."},
#     "done": true,
#     ...other metadata like token counts and timing
#   }
#
# The key field we care about is response["message"]["content"] — that's the
# model's actual answer text.


@app.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    async with httpx.AsyncClient() as client:
        ollama_response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": CHAT_MODEL,
                "messages": [{"role": "user", "content": request.message}],
                "stream": False,
            },
            timeout=120.0,
        )
        ollama_response.raise_for_status()
        data = ollama_response.json()

    return ChatResponse(
        response=data["message"]["content"],
        model=CHAT_MODEL,
    )


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

            async function sendQuestion() {
                const message = questionInput.value.trim();
                if (!message) return;

                // Show thinking state, disable input
                sendBtn.disabled = true;
                statusDiv.textContent = "Thinking...";
                responseDiv.style.display = "none";

                try {
                    const res = await fetch("/chat", {
                        method: "POST",
                        headers: {"Content-Type": "application/json"},
                        body: JSON.stringify({message}),
                    });

                    if (!res.ok) throw new Error("Request failed");

                    const data = await res.json();
                    responseDiv.textContent = data.response;
                    responseDiv.style.display = "block";
                    statusDiv.textContent = "Model: " + data.model;
                } catch (err) {
                    statusDiv.textContent = "Error: " + err.message;
                } finally {
                    sendBtn.disabled = false;
                    questionInput.focus();
                }
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
