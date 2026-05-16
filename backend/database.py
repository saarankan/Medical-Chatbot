from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from config import MONGODB_URI, DB_NAME, DOCS_COLLECTION, HISTORY_COLLECTION
from datetime import datetime

client   = MongoClient(MONGODB_URI)
db       = client[DB_NAME]
docs_col = db[DOCS_COLLECTION]
hist_col = db[HISTORY_COLLECTION]
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def retrieve_context(question: str, top_k: int = 3) -> str:
    """Embed question → search MongoDB → return top matching chunks."""
    query_embedding = embedder.encode(question).tolist()

    results = docs_col.aggregate([{
        "$vectorSearch": {
            "index":       "vector_index",
            "path":        "embedding",
            "queryVector": query_embedding,
            "numCandidates": 20,
            "limit":       top_k
        }
    }])

    chunks = [r["text"] for r in results]
    return "\n\n".join(chunks)

def save_message(session_id: str, role: str, content: str):
    hist_col.insert_one({
        "session_id": session_id,
        "role":       role,
        "content":    content,
        "timestamp":  datetime.utcnow()
    })

def get_history(session_id: str, limit: int = 6) -> list:
    messages = hist_col.find(
        {"session_id": session_id},
        sort=[("timestamp", -1)],
        limit=limit
    )
    return list(reversed(list(messages)))