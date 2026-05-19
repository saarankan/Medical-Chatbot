from dotenv import load_dotenv
import os

load_dotenv()

MONGODB_URI   = os.getenv("MONGODB_URI")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
DB_NAME       = "clinic"
DOCS_COLLECTION   = "clinic_docs"
HISTORY_COLLECTION = "chat_history"


import os

class Config:
    """
    Configuration class to manage file paths and settings for the application.
    """
    
    # --- PDF Configuration ---
    # Directory where source PDF documents are located.
    PDF_SOURCE_DIRECTORY: str = "data/pdfs"
    
    # --- doc Configuration ---
    # Directory where source documents are located.
    DOCUMENT_SOURCE_DIRECTORY: str = "data/Docs"
    # Directory where ChromaDB embeddings will be persisted.
    CHROMA_PERSIST_DIRECTORY: str = "docs/chroma"
    
    MONGODB_URI   = os.getenv("MONGODB_URI")
    GROQ_API_KEY  = os.getenv("GROQ_API_KEY")
    DB_NAME       = "clinic"
    DOCS_COLLECTION   = "clinic_docs"
    HISTORY_COLLECTION = "chat_history"



    # ---Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large" # "all-MiniLM-L6-v2"
    CHUNK_SIZE = 2028
    CHUNK_OVERLAP = 250

    def __init__(self):
        # Ensure the PDF source directory exists upon initialization.
        os.makedirs(self.PDF_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. PDF documents should be placed in '{self.PDF_SOURCE_DIRECTORY}'.")

        # Ensure the Docs source directory exists upon initialization.
        os.makedirs(self.DOCUMENT_SOURCE_DIRECTORY, exist_ok=True)
        print(f"Configuration loaded. Documents should be placed in '{self.DOCUMENT_SOURCE_DIRECTORY}'.")
# Create a global instance of the Config class for easy access.
# You can import `config` from this module in other files.
config = Config()
