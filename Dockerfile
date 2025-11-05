FROM python:3.10-slim

WORKDIR /app

# OS packages needed for LightGBM and wordcloud
RUN apt-get update && apt-get install -y \
    libgomp1 \
    build-essential \
    gcc \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY FastAPI/ /app/
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]