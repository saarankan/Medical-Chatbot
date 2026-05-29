import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import test_connection
from agents import run


# ── Small talk short-circuit — zero tokens used ──
SMALL_TALK = {
    "greetings": ["hi", "hello", "hey", "hii", "helo", "hai",
                  "good morning", "good afternoon", "good evening"],
    "thanks":    ["thanks", "thank you", "thank you!", "thanks!", "thx"],
    "bye":       ["bye", "goodbye", "see you", "good night"]
}

def get_small_talk_response(message: str) -> str | None:
    q = message.lower().strip()

    if q in SMALL_TALK["greetings"]:
        return "Hello! Welcome to our clinic. How can I help you today? You can ask about clinic hours, our doctors, services, or book an appointment."

    if q in SMALL_TALK["thanks"]:
        return "You're welcome! Is there anything else I can help you with?"

    if q in SMALL_TALK["bye"]:
        return "Goodbye! Take care and stay healthy."

    return None


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

    # ── Short-circuit small talk — never reaches LangGraph ──
    small_talk = get_small_talk_response(request.message)
    if small_talk:
        return {"response": small_talk, "session_id": session_id}

    try:
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