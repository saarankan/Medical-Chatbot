from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI   = os.getenv("MONGODB_URI")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
DB_NAME       = "Medical"
DOCS_COLLECTION   = "medical_docs"
HISTORY_COLLECTION = "chat_history"