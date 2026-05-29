from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI   = os.getenv("MONGODB_URI")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
MODEL        = os.getenv("MODEL", "llama-3.3-70b-versatile")  # default to llama-3.3 if MODEL not set
DB_NAME       = "clinic"
DOCS_COLLECTION   = "clinic_docs"
HISTORY_COLLECTION = "chat_history"
APPOINTMENTS_COLLECTION = os.getenv("APPOINTMENTS_COLLECTION", "appointments")
