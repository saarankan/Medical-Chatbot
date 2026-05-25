# 🏥 Clinic Medical Chatbot

A production-grade AI-powered medical assistant for a small clinic. Patients can ask questions about clinic services, doctors, and hours — and book appointments — entirely through natural conversation.

> **Status:** 🚧 In development — deployment in progress

---

## What it does

- **Medical FAQ** — answers patient questions from real clinic PDF documents using RAG (Retrieval-Augmented Generation). Never makes up information — if the answer is not in the clinic documents, it says "please call the clinic directly."
- **Appointment booking** — patients can check availability, book, cancel and view appointments through natural conversation
- **Conversation memory** — remembers context across messages in the same session
- **Safety by design** — refuses to diagnose conditions or recommend medications, by explicit system prompt rules

---

## Architecture

```
Patient (browser)
      ↓
  index.html + app.js        → plain HTML/CSS/JS frontend
      ↓  fetch() POST /chat
  main.py                    → FastAPI backend
      ↓
  agents.py                  → LangGraph supervisor
      ↓ classifies intent
  ┌───────────────────────────────────┐
  │  MEDICAL_FAQ  │      BOOKING      │
  │               │                   │
  │  rag.py       │  booking.py       │
  │  + MongoDB    │  + MongoDB        │
  │  vector search│  appointments     │
  │  + Groq LLM   │  collection       │
  └───────────────────────────────────┘
      ↓
  MongoDB Atlas              → clinic_docs + chat_history + appointments
  Groq API                   → Llama 3.3 70B inference
  LangSmith                  → LLM call tracing
```

### How the supervisor works

Every patient message goes through a **LangGraph supervisor** that classifies intent:

- `MEDICAL_FAQ` → routes to `rag_node` → retrieves relevant clinic document chunks from MongoDB Vector Search → sends to Groq for answer generation
- `BOOKING` → routes to `booking_node` → converses with patient to collect details → calls the right booking tool (check / create / cancel / list)

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM inference | Groq API — Llama 3.3 70B |
| Agent framework | LangGraph + LangChain |
| Backend | FastAPI + Uvicorn |
| Database | MongoDB Atlas |
| Vector search | MongoDB Atlas Vector Search |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| PDF parsing | PyMuPDF |
| Observability | LangSmith |
| Containerisation | Docker |
| Frontend | Plain HTML + CSS + vanilla JS |
| Package manager | [uv](https://docs.astral.sh/uv/) |

---

## Project structure

```
clinic-chatbot/
│
├── backend/
│   ├── main.py              # FastAPI app — /health and /chat endpoints
│   ├── agents.py            # LangGraph supervisor + routing logic
│   ├── rag.py               # RAG pipeline — retrieve context + generate answer
│   ├── booking.py           # 4 booking tools — check / create / cancel / list
│   ├── database.py          # MongoDB connection, vector search, chat history
│   ├── ingest.py            # PDF → chunks → embed → store in MongoDB (run once)
│   └── config.py            # Environment variables
|
│
├── docs/pdf                    # Clinic PDF documents (knowledge base)
│
├── frontend/
│   ├── index.html           # Chat UI
│   ├── style.css            # Styling — large fonts for elderly patients
│   └── app.js               # fetch() calls to backend, session management
│
├── .gitignore
└── README.md
├── pyproject.toml       # Project metadata + dependencies (uv)
├── uv.lock              # Locked dependency versions
├── Dockerfile
└── .env.example
```

---

## Local setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- MongoDB Atlas account (free tier)
- Groq API key (free — [console.groq.com](https://console.groq.com))
- LangSmith account (free — [smith.langchain.com](https://smith.langchain.com))

### Install uv (if not already installed)

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
# uv 0.x.x
```

---

### 1. Clone the repo

```bash
git clone https://github.com/Saarankan/clinic-chatbot.git
cd clinic-chatbot
```

### 2. Create virtual environment and install dependencies

```bash
cd backend

# create venv + install all dependencies in one command
uv sync
```

`uv sync` reads `pyproject.toml`, creates a `.venv` folder, and installs every dependency with exact versions from `uv.lock`. No manual activation needed for running scripts — `uv run` handles it.

### 3. Set up environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```bash
# .env

MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/?
GROQ_API_KEY=gsk_your_key_here

# LangSmith tracing (optional but recommended)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=clinic-chatbot-dev
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. Add your clinic documents

Drop clinic PDF files into the `docs/` folder:


```

### 5. Ingest documents into MongoDB

Reads all PDFs, chunks the text, embeds each chunk, and stores everything in MongoDB. Run once — and again whenever you add new documents.

```bash
uv run ingest.py
```

Expected output:

```
Found 4 PDF file(s)
Cleared 0 old chunks from MongoDB
Processing: services.pdf → Stored 12 chunks
Processing: doctors.pdf  → Stored 8 chunks
Processing: faq.pdf      → Stored 15 chunks
Processing: hours.pdf    → Stored 5 chunks

Ingestion complete — Total chunks stored: 40
Next step: create the Vector Search index in Atlas
```

### 6. Create MongoDB Atlas Vector Search index

After ingesting, create the vector search index manually in the Atlas dashboard (one-time step):

1. Go to [cloud.mongodb.com](https://cloud.mongodb.com) → your cluster → **Atlas Search**
2. Click **Create Search Index** → choose **Atlas Vector Search**
3. Select collection: `clinic` → `clinic_docs`
4. Choose **JSON Editor** and paste:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 384,
      "similarity": "cosine"
    }
  ]
}
```

5. Name the index: `vector_index`
6. Click **Create** — wait ~3 minutes for status to show **ACTIVE**

### 7. Test the connection and vector search

```bash
uv run database.py
```

```
TEST 1 — MongoDB connection
  MongoDB: Connected successfully
  Database : clinic
  Docs     : clinic_docs
  History  : chat_history

TEST 2 — Vector search
  RESULT: 3 chunk(s) returned
  Vector search working correctly.
  You are ready to run rag.py
```

### 8. Start the backend

```bash
uv run main.py
```

```
─────────────────────────────────
  Clinic Chatbot API starting...
─────────────────────────────────
  MongoDB: Connected successfully
─────────────────────────────────
  Server ready to accept requests
─────────────────────────────────
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 9. Open the frontend

Open `frontend/index.html` in VS Code → right-click → **Open with Live Server**

Chat UI opens at `http://127.0.0.1:5500`

> Make sure `API_URL` in `frontend/app.js` is set to `http://localhost:8000`

---

## Managing dependencies with uv

### Add a new package

```bash
# add a package and update pyproject.toml + uv.lock automatically
uv add package-name

# example
uv add httpx
```

### Remove a package

```bash
uv remove package-name
```

### Run any script

```bash
# uv run activates the venv automatically before running
uv run script-name.py

# examples
uv run main.py
uv run ingest.py
uv run database.py
uv run rag.py
uv run booking.py
uv run agents.py
```

### Update all dependencies

```bash
uv sync --upgrade
```

---

## Testing

### Test the RAG pipeline

```bash
uv run rag.py
```

Type questions in the terminal. Verify the bot answers from clinic documents and refuses to answer things not in the docs.

### Test the booking tools

```bash
uv run booking.py
```

Runs all 4 booking tools (check availability, create, list, cancel) in sequence. All 4 should pass.

### Test the supervisor routing

```bash
uv run agents.py
```

Tests that MEDICAL_FAQ questions route to the RAG agent and BOOKING requests route to the booking agent.

### Full end-to-end test

1. Start backend: `uv run main.py`
2. Open `frontend/index.html` with Live Server
3. Send a message in the browser
4. Check MongoDB Atlas → `chat_history` — two new documents should appear
5. Check LangSmith dashboard — a new trace should appear within 5 seconds

---

## Docker (local)

The Dockerfile uses `uv` for dependency installation — faster and more reproducible than pip inside containers.

```bash
cd backend

# build
docker build -t clinic-backend .

# run
docker run -p 8000:8000 --env-file .env clinic-backend
```

Test: open `http://localhost:8000/health` → should return `{"status": "ok"}`

### Dockerfile (uv-based)

```dockerfile
FROM python:3.11-slim

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# install dependencies with uv (no venv inside container — system install)
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
```

---

## Deployment

> **Coming soon** — deployment to Render (backend) + GitHub Pages (frontend)

Planned deployment stack:

| Service | Purpose | Cost |
|---|---|---|
| Render | FastAPI backend (Docker) | Free tier |
| GitHub Pages | Static HTML frontend | Free |
| MongoDB Atlas | Database + vector search | Free tier (512MB) |
| Groq API | LLM inference | Free tier |
| UptimeRobot | Keep Render warm (ping /health) | Free |
| LangSmith | LLM call tracing | Free tier (5k traces/month) |

After deployment, update one line in `frontend/app.js`:

```javascript
// change from:
const API_URL = "http://localhost:8000";

// to:
const API_URL = "https://your-app-name.onrender.com";
```

---

## MongoDB collections

| Collection | Purpose |
|---|---|
| `clinic_docs` | PDF chunks + 384-dim embeddings for vector search |
| `chat_history` | Patient conversation messages (role, content, timestamp) |
| `appointments` | Booking records (patient, doctor, date, time, status) |

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `MONGODB_URI` | ✅ | MongoDB Atlas connection string |
| `GROQ_API_KEY` | ✅ | Groq API key for LLM inference |
| `DB_NAME` | ❌ | Database name (default: `clinic`) |
| `DOCS_COLLECTION` | ❌ | Docs collection (default: `clinic_docs`) |
| `HISTORY_COLLECTION` | ❌ | History collection (default: `chat_history`) |
| `LANGCHAIN_TRACING_V2` | ❌ | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | ❌ | LangSmith API key |
| `LANGCHAIN_PROJECT` | ❌ | LangSmith project name |
| `LANGCHAIN_ENDPOINT` | ❌ | LangSmith endpoint URL |

---

## Adding new clinic documents

To update the chatbot's knowledge base:

1. Drop new PDF files into the `docs/` folder
2. Run `uv run ingest.py` — clears old chunks and re-ingests everything
3. No code changes needed — the chatbot immediately answers from the new content

---

## Safety and disclaimer

This chatbot is designed with patient safety as a priority:

- Answers **only** from provided clinic documents — no hallucination
- **Never** provides medical diagnoses
- **Never** recommends specific medications or dosages
- Always advises patients to call the clinic or visit a doctor for medical concerns
- Disclaimer banner is always visible on the frontend

---

## Author

**Saarankan Baskaran**
B.Sc Hons in Science and Technology (Mechatronics) — Uva Wellassa University of Sri Lanka

- GitHub: [github.com/Saarankan](https://github.com/Saarankan)
- LinkedIn: [linkedin.com/in/Saarankan-Baskaran](https://linkedin.com/in/Saarankan-Baskaran)
- Email: saarankan16@gmail.com

---

## License

MIT License — free to use, modify and distribute.