import fitz                          # PyMuPDF — reads PDF
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from config import MONGODB_URI, DB_NAME, DOCS_COLLECTION
import glob, os

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client   = MongoClient(MONGODB_URI)
col      = client[DB_NAME][DOCS_COLLECTION]

def extract_text_from_pdf(pdf_path: str) -> list:
    """Returns list of (page_num, text) tuples."""
    doc   = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append((i + 1, text))
    return pages

def chunk_text(text: str, size=200, overlap=40) -> list:
    """Split text into overlapping word chunks."""
    words  = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunk = " ".join(words[i:i + size])
        if chunk:
            chunks.append(chunk)
    return chunks

def ingest_pdf(pdf_path: str):
    filename = os.path.basename(pdf_path)
    print(f"Processing: {filename}")

    pages = extract_text_from_pdf(pdf_path)
    docs  = []

    for page_num, text in pages:
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = embedder.encode(chunk).tolist()
            docs.append({
                "text":      chunk,
                "embedding": embedding,
                "source":    filename,
                "page":      page_num
            })

    if docs:
        col.insert_many(docs)
        print(f"  Stored {len(docs)} chunks from {filename}")

def ingest_all():
    # clear old data before re-ingesting
    col.delete_many({})
    print("Cleared existing clinic_docs")

    for pdf_path in glob.glob("../docs/*.pdf"):
        ingest_pdf(pdf_path)

    print(f"\nDone. Total chunks: {col.count_documents({})}")

if __name__ == "__main__":
    ingest_all()