import asyncio
from groq import Groq
from database import retrieve_context, save_message, get_history
from config import GROQ_API_KEY


# ─────────────────────────────────────────────
#  HOW THIS FILE WORKS — READ FIRST
#
#  This is the brain of the chatbot.
#  Every time a patient sends a message,
#  this file does 3 things:
#
#  1. RETRIEVE  — asks database.py to find the
#                 most relevant clinic document
#                 chunks for this question
#
#  2. BUILD     — combines the question +
#                 retrieved chunks + chat history
#                 into one prompt for the AI
#
#  3. GENERATE  — sends the prompt to Groq API
#                 (which runs Llama 3.3 70B)
#                 and gets back an answer
#
#  The word RAG stands for:
#    Retrieval  — fetch relevant clinic docs
#    Augmented  — add them to the prompt
#    Generation — AI generates the answer
#
#  Without RAG, the AI would answer from its
#  general training data and might give wrong
#  information about your specific clinic.
#  With RAG, it answers from YOUR clinic's
#  actual documents.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  GROQ CLIENT
#  Groq is the service that runs the AI model.
#  Think of it like a phone — you call Groq,
#  give it a prompt, it gives back an answer.
#  The free tier is very generous for a clinic.
# ─────────────────────────────────────────────

groq_client = Groq(api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────
#  SYSTEM PROMPT
#  This is the instruction you give the AI
#  before every conversation. It tells the AI:
#    - who it is (clinic assistant)
#    - what it can answer (clinic info only)
#    - what it must never do (diagnose, prescribe)
#    - how to behave (polite, empathetic)
#
#  Think of it like a job description you give
#  to a new employee on their first day.
#  The AI reads this before every single message.
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
#
#  This is the only function main.py calls.
#  It takes the patient's question and returns
#  the chatbot's answer as a string.
#
#  Flow:
#    question + session_id
#        ↓
#    retrieve relevant clinic chunks (database.py)
#        ↓
#    get recent chat history (database.py)
#        ↓
#    build full prompt
#        ↓
#    send to Groq → get answer
#        ↓
#    save question + answer to MongoDB
#        ↓
#    return answer to main.py
# ─────────────────────────────────────────────

async def ask(question: str, session_id: str) -> str:
    """
    Takes a patient question and returns the chatbot answer.

    Args:
        question   : the patient's message text
        session_id : unique ID for this conversation
                     (used to load and save chat history)

    Returns:
        the chatbot's answer as a plain string
    """

    # ── STEP 1: RETRIEVE relevant clinic chunks ──
    # database.py searches MongoDB using vector search
    # and returns the 3 most relevant text chunks
    # from your clinic PDF documents

    context = await retrieve_context(question, top_k=3)

    # if no relevant chunks found, context will be empty string
    # the SYSTEM_PROMPT handles this — bot will say "call the clinic"
    if not context:
        context = "No specific clinic information found for this question."


    # ── STEP 2: GET recent chat history ──
    # this gives the AI short-term memory
    # so it remembers what was said earlier
    # in the same conversation
    #
    # Example without history:
    #   Patient: "Who are your doctors?"
    #   Bot: "Dr. Silva and Dr. Kumar"
    #   Patient: "What does the second one specialise in?"
    #   Bot: "I don't know who you mean"   ← bad, no memory
    #
    # Example with history:
    #   Patient: "Who are your doctors?"
    #   Bot: "Dr. Silva and Dr. Kumar"
    #   Patient: "What does the second one specialise in?"
    #   Bot: "Dr. Kumar specialises in Paediatrics"  ← good

    history = await get_history(session_id, limit=6)

    # format history as readable text for the prompt
    # turns [{"role": "user", "content": "hello"}] into "User: hello"
    if history:
        history_text = "\n".join([
            f"{'Patient' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history
        ])
    else:
        history_text = "No previous messages in this conversation."


    # ── STEP 3: BUILD the full prompt ──
    # we combine 3 things into one user message:
    #   a) clinic context  (retrieved chunks)
    #   b) recent history  (last 3 exchanges)
    #   c) current question
    #
    # The AI reads all 3 together to give a
    # contextually accurate and memory-aware answer

    user_prompt = f"""Clinic information (use this to answer):
{context}

---

Recent conversation history:
{history_text}

---

Patient's current question:
{question}"""


    # ── STEP 4: GENERATE answer with Groq ──
    # we send two messages to Groq:
    #   1. system message — the job description (SYSTEM_PROMPT)
    #   2. user message   — context + history + question
    #
    # Groq runs Llama 3.3 70B which is a very capable model
    # temperature=0.3 means "be fairly consistent and factual"
    #   (0.0 = robotic and repetitive, 1.0 = creative and unpredictable)
    # max_tokens=512 = maximum length of the answer

    try:
        response = groq_client.chat.completions.create(
            model       = "llama-3.3-70b-versatile",
            messages    = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            max_tokens  = 512,
            temperature = 0.3
        )

        answer = response.choices[0].message.content.strip()

    except Exception as e:
        # if Groq API fails for any reason, give a safe fallback
        # rather than crashing or showing a technical error to the patient
        print(f"Groq API error: {e}")
        answer = "I'm sorry, I'm having trouble responding right now. Please call the clinic directly for assistance."


    # ── STEP 5: SAVE to MongoDB ──
    # save both the patient's question and the bot's answer
    # so get_history() can return them in future messages

    await save_message(session_id, "user",      question)
    await save_message(session_id, "assistant", answer)


    return answer


# ─────────────────────────────────────────────
#  LOCAL TEST
#  Run this file directly to test the full
#  RAG pipeline without needing the frontend.
#
#  Usage:
#    python backend/rag.py
#
#  It will ask you to type questions in the
#  terminal and print the bot's answers.
#  This is the fastest way to check that
#  everything works before building the API.
# ─────────────────────────────────────────────

if __name__ == "__main__":

    async def run_test():
        print("\n─────────────────────────────────")
        print("  RAG pipeline test")
        print("  Type a question, press Enter")
        print("  Type 'quit' to stop")
        print("─────────────────────────────────\n")

        # use a fixed session ID for testing
        test_session = "test-session-001"

        while True:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit", "q"]:
                print("Stopped.")
                break

            print("Bot: thinking...", end="\r")

            answer = await ask(question, test_session)

            print(f"Bot: {answer}\n")

    asyncio.run(run_test())