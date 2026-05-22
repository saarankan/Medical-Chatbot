import asyncio
from groq import Groq
from database import retrieve_context, save_message, get_history
from config import GROQ_API_KEY


groq_client = Groq(api_key=GROQ_API_KEY)


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
    
    context = await retrieve_context(question, top_k=3)

    # if no relevant chunks found, context will be empty string
    # the SYSTEM_PROMPT handles this — bot will say "call the clinic"
    if not context:
        context = "No specific clinic information found for this question."


    # ── STEP 2: GET recent chat history ──

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
    
    user_prompt = f"""Clinic information (use this to answer):
{context}

---

Recent conversation history:
{history_text}

---

Patient's current question:
{question}"""


    # ── STEP 4: GENERATE answer with Groq ──
    
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
  
    await save_message(session_id, "user",      question)
    await save_message(session_id, "assistant", answer)


    return answer



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