import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import test_connection
from rag import ask



@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    print("\n─────────────────────────────────")
    print("  Clinic Chatbot API starting...")
    print("─────────────────────────────────")

    try:
        await test_connection()
        print("  Server ready to accept requests")
        print("─────────────────────────────────\n")
    except SystemExit:
        
        print("  WARNING: MongoDB connection failed")
        print("  Check your MONGODB_URI in environment variables")
        print("─────────────────────────────────\n")

    yield   # server runs here — everything above is startup

    # ── shutdown ──
    # nothing special needed on shutdown for this project
    print("\nServer shutting down.")


# ─────────────────────────────────────────────
#  CREATE THE FASTAPI APP
#  lifespan= wires up our startup/shutdown logic
# ─────────────────────────────────────────────

app = FastAPI(
    title       = "Clinic Chatbot API",
    description = "Medical clinic assistant powered by RAG",
    version     = "1.0.0",
    lifespan    = lifespan
)



app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # TODO: change to your GitHub Pages URL after deployment
    allow_credentials = True,
    allow_methods     = ["*"],   # allow GET, POST, OPTIONS etc.
    allow_headers     = ["*"],   # allow Content-Type, Authorization etc.
)



class ChatRequest(BaseModel):
    message:    str             # required — the patient's question
    session_id: str | None = None   # optional — if None we generate one




@app.get("/health")
async def health():
    return {"status": "ok"}



@app.post("/chat")
async def chat(request: ChatRequest):

    # ── validate the message is not empty ──
    if not request.message.strip():
        raise HTTPException(
            status_code = 400,
            detail      = "Message cannot be empty"
        )

    # ── generate session_id if not provided ──
    # uuid4() creates a random unique ID like: "a3f8c2d1-4b5e-..."
    # this is the patient's conversation ID
    session_id = request.session_id or str(uuid.uuid4())

    # ── call rag.ask() — this does all the work ──
    # retrieve context → build prompt → call Groq → save history
    try:
        response = await ask(
            question   = request.message.strip(),
            session_id = session_id
        )
    except Exception as e:
        # catch any unexpected errors from rag.py
        # log it server-side but send a safe message to the patient
        print(f"Error in /chat: {e}")
        raise HTTPException(
            status_code = 500,
            detail      = "Something went wrong. Please try again or call the clinic directly."
        )

    # ── return answer + session_id to frontend ──
    return {
        "response":   response,
        "session_id": session_id    # frontend stores this for next message
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True      # remove this on production
    )