import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import test_connection

# ── ONE LINE CHANGED from the original main.py ──
# Before: from rag import ask
# After:  from agents import run
# Everything else is identical.
from agents import run


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n─────────────────────────────────")
    print("  Clinic Chatbot API starting...")
    print("─────────────────────────────────")
    try:
        await test_connection()
        print("  Server ready to accept requests")
        print("─────────────────────────────────\n")
    except SystemExit:
        print("  WARNING: MongoDB connection failed")
        print("─────────────────────────────────\n")
    yield
    print("\nServer shutting down.")


app = FastAPI(
    title       = "Clinic Chatbot API",
    description = "Medical clinic assistant — RAG + booking agent",
    version     = "2.0.0",
    lifespan    = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class ChatRequest(BaseModel):
    message    : str
    session_id : str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        # ── now calls agents.run() instead of rag.ask() ──
        # agents.run() routes to rag_node or booking_node
        # depending on the patient's intent.
        # The response is always a plain string — same as before.
        response = await run(
            message    = request.message.strip(),
            session_id = session_id
        )

    except Exception as e:
        print(f"Error in /chat: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = "Something went wrong. Please try again or call the clinic directly."
        )

    return {
        "response":   response,
        "session_id": session_id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)