import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from database import retrieve_context, save_message, get_history
from config import GROQ_API_KEY


# ─────────────────────────────────────────────
#  WHAT CHANGED FROM THE ORIGINAL rag.py
#
#  Before: used  groq.Groq()  directly
#  After:  uses  langchain_groq.ChatGroq()
#
#  ChatGroq is a thin wrapper around the same
#  Groq API. The model, temperature, and
#  max_tokens are identical.
#
#  The only difference: LangChain automatically
#  reads your LANGCHAIN_* environment variables
#  and sends a trace to LangSmith for every
#  single LLM call. You see the full prompt,
#  response, token usage, and latency in the
#  LangSmith dashboard.
#
#  No other files need to change.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  LLM CLIENT
#  Same model, same settings as before.
#  Just using the LangChain wrapper now.
# ─────────────────────────────────────────────

llm = ChatGroq(
    model       = "llama-3.3-70b-versatile",
    api_key     = GROQ_API_KEY,
    max_tokens  = 512,
    temperature = 0.3
)


# ─────────────────────────────────────────────
#  SYSTEM PROMPT — unchanged
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful and friendly assistant for a medical clinic.

Your job:
- Answer patient questions using ONLY the clinic information provided to you
- Help patients understand clinic services, doctors, and opening hours
- Be polite, clear, and empathetic at all times

Your strict rules:
- If the answer is NOT in the provided clinic information, say:
  "I don't have that information. Please call the clinic directly."
- NEVER provide medical diagnoses
- NEVER recommend specific medications or dosages
- NEVER tell a patient whether their symptoms are serious or not
- NEVER make up information that is not in the clinic documents

Important disclaimer to add when relevant:
"This chatbot provides general clinic information only and cannot replace a doctor's advice."
"""


# ─────────────────────────────────────────────
#  MAIN FUNCTION — ask()
#  Identical logic to before.
#  LangSmith tracing happens automatically
#  inside llm.invoke() — nothing extra needed.
# ─────────────────────────────────────────────

async def ask(question: str, session_id: str) -> str:

    # STEP 1 — retrieve relevant clinic chunks
    context = await retrieve_context(question, top_k=3)
    if not context:
        context = "No specific clinic information found for this question."

    # STEP 2 — get recent chat history for memory
    history = await get_history(session_id, limit=6)
    if history:
        history_text = "\n".join([
            f"{'Patient' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])
    else:
        history_text = "No previous messages in this conversation."

    # STEP 3 — build the prompt
    user_prompt = f"""Clinic information (use this to answer):
{context}

---

Recent conversation history:
{history_text}

---

Patient's current question:
{question}"""

    # STEP 4 — call Groq via LangChain
    # LangSmith automatically traces this call:
    #   → records the full prompt sent
    #   → records the full response received
    #   → records token usage and latency
    #   → saves it to your LangSmith project
    try:
        loop     = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])
        )
        answer = response.content.strip()

    except Exception as e:
        print(f"LLM error: {e}")
        answer = (
            "I'm sorry, I'm having trouble responding right now. "
            "Please call the clinic directly for assistance."
        )

    # STEP 5 — save to MongoDB
    await save_message(session_id, "user",      question)
    await save_message(session_id, "assistant", answer)

    return answer


# ─────────────────────────────────────────────
#  LOCAL TEST
#  python backend/rag.py
# ─────────────────────────────────────────────

if __name__ == "__main__":

    async def run_test():
        print("\n─────────────────────────────────")
        print("  RAG + LangSmith tracing test")
        print("  Type a question, press Enter")
        print("  Type 'quit' to stop")
        print("─────────────────────────────────\n")

        test_session = "test-session-langsmith"

        while True:
            question = input("You: ").strip()
            if not question:     continue
            if question.lower() in ["quit", "exit", "q"]: break

            print("Bot: thinking...", end="\r")
            answer = await ask(question, test_session)
            print(f"Bot: {answer}\n")
            print("→ Check smith.langchain.com for the trace\n")

    asyncio.run(run_test())