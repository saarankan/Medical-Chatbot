import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from sentence_transformers import SentenceTransformer
from config import MONGODB_URI, DB_NAME, DOCS_COLLECTION, HISTORY_COLLECTION


# ─────────────────────────────────────────────
#  1. CONNECTION
#  motor is the async version of pymongo.
#  We create one client and reuse it everywhere.
#  Think of it like opening one door to MongoDB
#  and keeping it open the whole time the app runs.
# ─────────────────────────────────────────────

client = AsyncIOMotorClient(MONGODB_URI)
db     = client[DB_NAME]

docs_col = db[DOCS_COLLECTION]     # stores: clinic PDF chunks + embeddings
hist_col = db[HISTORY_COLLECTION]  # stores: patient chat messages


# ─────────────────────────────────────────────
#  2. EMBEDDING MODEL
#  This converts text → 384 numbers (a vector).
#  We use the same model in both ingest.py and
#  here so the numbers are in the same "language".
#  If you used model A to store, use model A to search.
# ─────────────────────────────────────────────

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ─────────────────────────────────────────────
#  3. CONNECTION TEST
#  Call this once when the app starts.
#  If it fails → something is wrong with your
#  MONGODB_URI or your internet connection.
#  Fix it before writing anything else.
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  4. VECTOR SEARCH  (used by rag.py)
#  Steps:
#    a) embed the patient's question → 384 numbers
#    b) ask MongoDB to find the 3 clinic chunks
#       whose embeddings are closest in meaning
#    c) join those chunks into one block of text
#       that rag.py will give to Groq as context
#
#  This only works AFTER you:
#    - ran ingest.py   (chunks are in MongoDB)
#    - created the Atlas Vector Search index
#      named "vector_index" on the embedding field
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  5. CHAT HISTORY  (used by main.py and rag.py)
#
#  Every message the patient sends and every
#  response the bot gives is stored here.
#  session_id ties messages to one conversation.
#  A patient keeps the same session_id until
#  they close the browser (stored in localStorage).
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
#  6. QUICK LOCAL TEST
#  Run this file directly:  python database.py
#  It will test the connection and print the result.
#  You should see "Connected successfully" before
#  moving on to write ingest.py or rag.py.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    async def run_test():
        print("\nTesting MongoDB connection...")
        await test_connection()
        print("\nConnection test passed.")
        print("You can now run ingest.py to load your clinic documents.")

    asyncio.run(run_test())