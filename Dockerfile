FROM python:3.11-slim

WORKDIR /app

# System deps (keep minimal but practical for matplotlib/wordcloud)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libfreetype6 \
    libjpeg-dev \
    libpng-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8003

# Run FastAPI app
CMD ["python", "-m", "uvicorn", "FastAPI.app:app", "--host", "0.0.0.0", "--port", "8003"]


