# ── STEP 1: Choose the base image ──
FROM python:3.11-slim


# ── STEP 2: Set the working directory ──
WORKDIR /app


# ── STEP 3: Copy requirements.txt first ──
COPY requirements.txt .


# ── STEP 4: Install Python dependencies ──
RUN pip install --no-cache-dir -r requirements.txt


# ── STEP 5: Copy the rest of your code ──
COPY . .


# ── STEP 6: Tell Docker which port the app uses ──
EXPOSE 8000


# ── STEP 7: The command to start the app ──
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]