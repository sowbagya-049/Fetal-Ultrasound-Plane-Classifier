# ── Fetal Ultrasound Classifier — Docker Image ────────────────────────────────
# Base image: slim Python 3.10 (CPU-only PyTorch)
FROM python:3.10-slim

# Metadata
LABEL maintainer="Sowbagya VS"
LABEL description="Fetal Ultrasound Plane Classifier – EfficientNet-B0"

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies FIRST (Docker layer cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY enhanced_app.py .
COPY api.py .
COPY best_efficientnet.pth .

# Expose ports
EXPOSE 8501   
# Streamlit default port
EXPOSE 8000   
# FastAPI default port

# ── Default CMD: run Streamlit app ────────────────────────────────────────────
CMD ["streamlit", "run", "enhanced_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]

# ── To run FastAPI instead, override CMD: ─────────────────────────────────────
# docker run -p 8000:8000 fetal-classifier \
#   uvicorn api:app --host 0.0.0.0 --port 8000