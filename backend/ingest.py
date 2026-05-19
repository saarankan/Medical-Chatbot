import asyncio
import os
import glob
import fitz                          # PyMuPDF — reads PDF files
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
from config import MONGODB_URI, DB_NAME, DOCS_COLLECTION


# ─────────────────────────────────────────────
#  HOW THIS FILE WORKS — READ FIRST
#
#  You run this file ONCE before starting the chatbot.
#  It does 4 things in order:
#
#  1. READ    — opens every PDF in your docs/ folder
#  2. CHUNK   — cuts the text into small 200-word pieces
#  3. EMBED   — converts each piece into 384 numbers
#               that represent its meaning
#  4. STORE   — saves everything into MongoDB
#
#  After this runs, your chatbot can search those
#  chunks when a patient asks a question.
#
#  Run it again any time you add new PDF files.
#  It clears old data first so you never get duplicates.
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
#  SETUP
#  Connect to MongoDB and load the embedding model.
#  Same model as database.py — this is important.
#  If you change the model here, change it there too.
# ─────────────────────────────────────────────

client   = AsyncIOMotorClient(MONGODB_URI)
db       = client[DB_NAME]
docs_col = db[DOCS_COLLECTION]

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# where your PDF files live
DOCS_FOLDER = os.path.join(os.path.dirname(__file__), "..", "docs")


# ─────────────────────────────────────────────
#  STEP 1 — READ PDF
#  PyMuPDF (fitz) opens the PDF and extracts
#  the text from every page.
#
#  Returns a list of (page_number, text) pairs.
#  Example:
#    [(1, "Dr. Silva is available on..."),
#     (2, "The clinic opens at 8am...")]
# ─────────────────────────────────────────────

def read_pdf(pdf_path: str) -> list:
    """
    Opens a PDF and extracts text from each page.

    Args:
        pdf_path: full path to the PDF file

    Returns:
        list of (page_number, page_text) tuples
        skips blank pages automatically
    """
    pages = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"  Could not open {pdf_path}: {e}")
        return []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        # skip blank or near-blank pages
        if len(text) < 50:
            continue

        pages.append((page_num + 1, text))   # page_num is 0-indexed, +1 for human-readable

    doc.close()
    return pages


# ─────────────────────────────────────────────
#  STEP 2 — CHUNK TEXT
#  We cannot send a whole PDF to the AI at once —
#  it is too long and expensive.
#
#  Instead we cut it into small overlapping pieces.
#
#  Example with size=10 words, overlap=3 words:
#    Text: "the cat sat on the mat in the sun today"
#    Chunk 1: "the cat sat on the mat in"       (words 0-9)
#    Chunk 2: "in the sun today"                (words 7-10)
#             ^^^
#             overlap — repeated words help keep context
#             across chunk boundaries
#
#  Why overlap? If an important sentence happens
#  to be split between chunk 1 and chunk 2,
#  the overlap means it appears (at least partly)
#  in both chunks so it is never fully lost.
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list:
    """
    Splits text into overlapping word chunks.

    Args:
        text       : the raw text to split
        chunk_size : how many words per chunk (200 is a good default)
        overlap    : how many words to repeat between chunks (40 is good)

    Returns:
        list of text chunk strings
    """
    words  = text.split()      # split on whitespace — gives a list of words
    chunks = []
    i      = 0

    while i < len(words):
        # take chunk_size words starting at position i
        chunk_words = words[i : i + chunk_size]
        chunk_text  = " ".join(chunk_words)

        # only keep chunks with meaningful content (at least 30 words)
        if len(chunk_words) >= 30:
            chunks.append(chunk_text)

        # move forward by (chunk_size - overlap) words
        # this is what creates the overlap
        i += chunk_size - overlap

    return chunks


# ─────────────────────────────────────────────
#  STEP 3 — EMBED TEXT
#  The embedding model reads a chunk of text
#  and converts it into 384 numbers.
#
#  These numbers capture the MEANING of the text,
#  not just the words. So:
#    "clinic is open Monday"
#    "we are available on Mondays"
#  ...will have similar numbers even though
#  the words are different.
#
#  This is what makes semantic search work —
#  a patient can ask in any wording and still
#  find the right answer.
# ─────────────────────────────────────────────

def embed(text: str) -> list:
    """
    Converts a text string into a vector of 384 numbers.

    Args:
        text: the chunk of text to embed

    Returns:
        list of 384 floats — the vector
    """
    vector = embedder.encode(text)
    return vector.tolist()    # convert numpy array → plain Python list for MongoDB


# ─────────────────────────────────────────────
#  STEP 4 — STORE IN MONGODB
#  Each chunk is stored as one document in MongoDB.
#  The document contains:
#    - text      : the actual words (for the AI to read)
#    - embedding : the 384 numbers (for searching)
#    - source    : which PDF file it came from
#    - page      : which page of that PDF
#
#  We batch insert for speed — one insert_many()
#  is much faster than many insert_one() calls.
# ─────────────────────────────────────────────

async def store_chunks(chunks_data: list) -> int:
    """
    Saves a batch of chunks to MongoDB.

    Args:
        chunks_data: list of dicts, each with
                     text, embedding, source, page

    Returns:
        number of chunks stored
    """
    if not chunks_data:
        return 0

    result = await docs_col.insert_many(chunks_data)
    return len(result.inserted_ids)


# ─────────────────────────────────────────────
#  MAIN FUNCTION — ties all 4 steps together
#  This is what actually runs when you call
#  python ingest.py from the terminal
# ─────────────────────────────────────────────

async def ingest_all():
    """
    Main ingestion pipeline.
    Reads all PDFs from docs/ folder,
    chunks + embeds + stores every page,
    then prints a summary.
    """

    # ── find all PDF files in the docs/ folder ──
    pdf_pattern = os.path.join(DOCS_FOLDER, "*.pdf")
    pdf_files   = glob.glob(pdf_pattern)

    if not pdf_files:
        print("\n No PDF files found.")
        print(f" Put your clinic PDF files in: {os.path.abspath(DOCS_FOLDER)}")
        print(" Then run this script again.\n")
        return

    print(f"\n Found {len(pdf_files)} PDF file(s):")
    for f in pdf_files:
        print(f"   - {os.path.basename(f)}")

    # ── clear old data before re-ingesting ──
    # this prevents duplicate chunks if you run the script twice
    deleted = await docs_col.delete_many({})
    print(f"\n Cleared {deleted.deleted_count} old chunks from MongoDB")
    print(" Starting ingestion...\n")

    total_chunks = 0

    # ── process each PDF one by one ──
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        print(f" Processing: {filename}")

        # STEP 1 — read pages from PDF
        pages = read_pdf(pdf_path)

        if not pages:
            print(f"   Skipped — no readable text found")
            continue

        print(f"   Read {len(pages)} page(s)")

        chunks_data  = []   # will hold all chunks from this PDF
        chunk_count  = 0

        # STEP 2 + 3 — chunk and embed each page
        for page_num, page_text in pages:
            chunks = chunk_text(page_text)

            for chunk in chunks:
                # STEP 3 — embed the chunk
                vector = embed(chunk)

                # build the MongoDB document for this chunk
                chunks_data.append({
                    "text":      chunk,       # the actual text (AI reads this)
                    "embedding": vector,      # 384 numbers (used for searching)
                    "source":    filename,    # which PDF it came from
                    "page":      page_num     # which page number
                })

                chunk_count += 1

        # STEP 4 — store all chunks from this PDF in one batch
        stored = await store_chunks(chunks_data)
        total_chunks += stored
        print(f"   Stored {stored} chunks")

    # ── final summary ──
    print("\n─────────────────────────────────")
    print(f"  Ingestion complete")
    print(f"  Total chunks stored: {total_chunks}")
    print(f"  Collection: {DB_NAME}.{DOCS_COLLECTION}")
    print("─────────────────────────────────")
    print("\n Next step:")
    print("  Go to MongoDB Atlas dashboard")
    print("  Create a Vector Search index named 'vector_index'")
    print("  on the field 'embedding' with dimension 384")
    print("  Then you can run rag.py\n")


# ─────────────────────────────────────────────
#  RUN
#  asyncio.run() is needed because our functions
#  are async (they use await for MongoDB calls).
#  Think of it as the "starter" for async code.
# ─────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(ingest_all())