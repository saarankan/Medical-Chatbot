import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from config import MONGODB_URI, DB_NAME, DOCS_COLLECTION, HISTORY_COLLECTION

#  1. CONNECTION

client = AsyncIOMotorClient(MONGODB_URI)
db     = client[DB_NAME]

docs_col = db[DOCS_COLLECTION]     # stores: clinic PDF chunks + embeddings
hist_col = db[HISTORY_COLLECTION]  # stores: patient chat messages

#  2. EMBEDDING MODEL

embedder = SentenceTransformer("all-MiniLM-L6-v2")


#  3. CONNECTION TEST

async def test_connection():
    """
    Pings MongoDB to confirm the connection works.
    Prints Connected or an error message.
    Call this at the top of main.py on startup.
    """
    try:
        await client.admin.command("ping")
        print("─────────────────────────────────")
        print("  MongoDB: Connected successfully")
        print(f"  Database : {DB_NAME}")
        print(f"  Docs     : {DOCS_COLLECTION}")
        print(f"  History  : {HISTORY_COLLECTION}")
        print("─────────────────────────────────")
    except Exception as e:
        print("─────────────────────────────────")
        print("  MongoDB: CONNECTION FAILED")
        print(f"  Error: {e}")
        print("  Fix your MONGODB_URI in .env")
        print("─────────────────────────────────")
        raise SystemExit(1)   # stop the app — do not proceed if DB is down


#  4. VECTOR SEARCH  (used by rag.py)

async def retrieve_context(question: str, top_k: int = 3) -> str:
    """
    Converts the question to a vector, searches MongoDB
    for the most relevant clinic document chunks,
    and returns them joined as a single string.

    Args:
        question : the patient's raw question text
        top_k    : how many chunks to retrieve (3 is a good default)

    Returns:
        A string of relevant clinic information for rag.py to use.
        Returns empty string if nothing is found.
    """

    # step a — embed the question into 384 numbers
    # run_in_executor prevents this CPU-heavy operation
    # from blocking the async event loop
    loop = asyncio.get_event_loop()
    query_vector = await loop.run_in_executor(
        None,
        lambda: embedder.encode(question).tolist()
    )

    # step b — search MongoDB using the Atlas Vector Search index
    # $vectorSearch finds chunks whose embeddings are closest
    # to the question embedding using cosine similarity
    pipeline = [
        {
            "$vectorSearch": {
                "index":         "vector_index",   # name you gave when creating the index in Atlas
                "path":          "embedding",       # the field in clinic_docs that holds the vector
                "queryVector":   query_vector,
                "numCandidates": 20,                # check 20 candidates, return top_k
                "limit":         top_k
            }
        },
        {
            # only return the text and source — we don't need the huge embedding array back
            "$project": {
                "text":   1,
                "source": 1,
                "page":   1,
                "_id":    0
            }
        }
    ]

    cursor = docs_col.aggregate(pipeline)
    results = await cursor.to_list(length=top_k)

    if not results:
        # no matching chunks found — rag.py will tell the patient to call the clinic
        return ""

    # step c — join the chunks into one readable block of text
    chunks = [r["text"] for r in results]
    return "\n\n".join(chunks)


#  5. CHAT HISTORY  (used by main.py and rag.py)

async def save_message(session_id: str, role: str, content: str) -> None:
    """
    Saves one message to chat_history in MongoDB.

    Args:
        session_id : unique ID for this patient's conversation
        role       : "user" (patient) or "assistant" (bot)
        content    : the actual message text
    """
    await hist_col.insert_one({
        "session_id": session_id,
        "role":       role,
        "content":    content,
        "timestamp":  datetime.utcnow()
    })


async def get_history(session_id: str, limit: int = 6) -> list:
    """
    Gets the last N messages for a session.
    rag.py uses this to give the AI short-term memory —
    so if the patient says 'what about the second doctor?'
    the bot remembers what the first doctor question was.

    Args:
        session_id : the patient's session ID
        limit      : how many recent messages to fetch (6 = 3 exchanges)

    Returns:
        A list of message dicts ordered oldest → newest.
        Example: [{"role": "user", "content": "..."}, ...]
    """
    cursor = hist_col.find(
        {"session_id": session_id},
        sort=[("timestamp", -1)],   # get newest first
        limit=limit
    )
    messages = await cursor.to_list(length=limit)

    # reverse so oldest message comes first (natural conversation order)
    messages.reverse()

    return [{"role": m["role"], "content": m["content"]} for m in messages]


#  6. QUICK LOCAL TEST

if __name__ == "__main__":
    async def run_test():
        print("\nTesting MongoDB connection...")
        await test_connection()
        print("\nConnection test passed.")
        print("You can now run ingest.py to load your clinic documents.")

    asyncio.run(run_test())