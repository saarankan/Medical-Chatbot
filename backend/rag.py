from groq import Groq
from database import retrieve_context, get_history
from config import GROQ_API_KEY

groq = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant for a medical clinic.
Answer using ONLY the provided clinic information.
If the answer is not in the context, say: 'Please call the clinic directly.'
Never give medical diagnoses or prescriptions. Be polite and empathetic."""

def ask(question: str, session_id: str) -> str:
    # 1. retrieve relevant chunks from MongoDB
    context = retrieve_context(question)

    # 2. get recent chat history for conversational memory
    history = get_history(session_id)
    history_text = "\n".join(
        [f"{m['role'].title()}: {m['content']}" for m in history]
    )

    # 3. build prompt with context + memory
    user_prompt = f"""Clinic information:
{context}

Previous conversation:
{history_text}

Patient question: {question}"""

    response = groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=512,
        temperature=0.3
    )
    return response.choices[0].message.content