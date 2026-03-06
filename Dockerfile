# ═══════════════════════════════════════════════════
#  CDSS Backend — Production Dockerfile
# ═══════════════════════════════════════════════════
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install system-level dependencies ──
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libglib2.0-0 libgl1 && \
    rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies (cached layer) ──
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application source ──
COPY api.py .
COPY Models/ ./Models/
COPY Dataset/data.csv ./Dataset/data.csv
COPY Dataset/images/ ./Dataset/images/

# ── Expose API port ──
EXPOSE 8000

# ── Run the FastAPI server ──
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
