from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI   = os.getenv("MONGODB_URI")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
DB_NAME       = "clinic"
DOCS_COLLECTION   = "clinic_docs"
HISTORY_COLLECTION = "chat_history"
APPOINTMENTS_COLLECTION = os.getenv("APPOINTMENTS_COLLECTION", "appointments")
