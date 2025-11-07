import io
import re
import joblib
import mlflow
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from wordcloud import WordCloud

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import nltk
import os


# NLTK setup
nltk_packages = ["stopwords", "wordnet"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# CONFIG
# ----
MLFLOW_TRACKING_URI = "http://ec2-3-135-238-101.us-east-2.compute.amazonaws.com:5000/"
REGISTERED_MODEL_NAME = "yt_chrome_plugin_model"
REGISTERED_MODEL_VERSION = "3" 
VECTORIZER_PATH = "s3://project1-mlflow-bucket/1/0c56c31ba9c54ebb8c15044084e31fb4/artifacts/tfidf_vectorizer.pkl"
SKIP_MODEL_LOADING = os.getenv("SKIP_MODEL_LOADING") == "1"

STOP_WORDS = set(stopwords.words("english")) - {"not", "but", "however", "no", "yet"}

lemmatizer = WordNetLemmatizer()
sentiment_label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}


# TEXT PREPROCESS
def preprocess_comment(comment: str) -> str:
  
    try:
        text = comment.lower().strip()
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[^A-Za-z0-9\s!?.,]", "", text)

        tokens = [w for w in text.split() if w not in STOP_WORDS]
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

        return " ".join(tokens)
    except Exception as e:
        print(f"[preprocess_comment] ERROR: {e}")
        # if something weird happens, just return the raw string instead of crashing
        return comment


# MODEL / VECTORIZER LOADING
def load_model_from_registry(model_name: str, model_version: str):

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)


def load_vectorizer(path: str):

    if path.startswith("s3://"):
        # download_artifacts() returns a local file path
        local_path = mlflow.artifacts.download_artifacts(path)
        return joblib.load(local_path)
    else:
        return joblib.load(path)


# APP INIT
app = FastAPI(
    title="YouTube Comment Sentiment API",
    description="Predict sentiment, generate charts, wordclouds, and trend graphs.",
    version="1.0.0",
)

# CORS so browser extensions/frontends can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not SKIP_MODEL_LOADING:
    try:
        print(f"[startup] Loading model {REGISTERED_MODEL_NAME} version {REGISTERED_MODEL_VERSION} ...")
        model = load_model_from_registry(REGISTERED_MODEL_NAME, REGISTERED_MODEL_VERSION)

        print(f"[startup] Loading vectorizer from {VECTORIZER_PATH} ...")
        vectorizer = load_vectorizer(VECTORIZER_PATH)

        print("[startup] Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"[startup] ERROR loading model/vectorizer: {e}")
        model = None
        vectorizer = None
else:
    # super-light fake implementations so endpoints work in CI
    class _FakeVec:
        def transform(self, arr): 
            # your routes call .transform(preprocessed_list)
            # return something iterable with same length
            return arr

    class _FakeModel:
        def predict(self, X):
            # return one label per item
            return [1 for _ in X]  # pretend everything is positive

    model = _FakeModel()
    vectorizer = _FakeVec()
    print("[startup] SKIP_MODEL_LOADING=1 -> using fake model/vectorizer.")


# ROUTES
@app.get("/")
def home():
    return {"message": "FastAPI sentiment service is live."}



@app.post("/predict_with_timestamps")
def predict_with_timestamps(payload: dict):

    if "comments" not in payload:
        raise HTTPException(status_code=400, detail="No comments provided")

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    try:
        rows = payload["comments"]
        comments = [row["text"] for row in rows]
        timestamps = [row["timestamp"] for row in rows]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)

        raw_preds = model.predict(transformed)
        preds = [str(p) for p in np.array(raw_preds).tolist()]

        response = []
        for c, s, t in zip(comments, preds, timestamps):
            response.append({
                "comment": c,
                "sentiment": s,
                "timestamp": t
            })

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict")
def predict(payload: dict):

    if "comments" not in payload:
        raise HTTPException(status_code=400, detail="No comments provided")

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    try:
        comments = payload["comments"]

        preprocessed = [preprocess_comment(c) for c in comments]
        transformed = vectorizer.transform(preprocessed)

        raw_preds = model.predict(transformed)
        preds = [int(p) for p in np.array(raw_preds).tolist()]

        # build per-comment result
        detailed = []
        for c, s in zip(comments, preds):
            detailed.append({
                "comment": c,
                "sentiment": s
            })

        # build counts
        sentiment_counts = {
            "1": sum(1 for x in preds if x == 1),
            "0": sum(1 for x in preds if x == 0),
            "-1": sum(1 for x in preds if x == -1),
        }

        return JSONResponse(content={
            "predictions": detailed,
            "sentiment_counts": sentiment_counts
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/generate_chart")
def generate_chart(payload: dict):

    try:
        sentiment_counts = payload.get("sentiment_counts")
        if not sentiment_counts:
            raise HTTPException(status_code=400, detail="No sentiment counts provided")

        labels = ["Positive", "Neutral", "Negative"]
        sizes = [
            int(sentiment_counts.get("1", 0)),
            int(sentiment_counts.get("0", 0)),
            int(sentiment_counts.get("-1", 0)),
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ["#36A2EB", "#C9CBCF", "#FF6384"]  # blue, gray, red

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=140,
            textprops={"color": "w"},
        )
        plt.axis("equal")

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG", transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chart generation failed: {str(e)}"
        )


@app.post("/generate_wordcloud")
def generate_wordcloud(payload: dict):

    try:
        comments = payload.get("comments")
        if not comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        preprocessed_comments = [preprocess_comment(c) for c in comments]
        text = " ".join(preprocessed_comments)

        wc = WordCloud(
            width=800,
            height=400,
            background_color="black",
            colormap="Blues",
            stopwords=set(stopwords.words("english")),
            collocations=False,
        ).generate(text)

        img_io = io.BytesIO()
        wc.to_image().save(img_io, format="PNG")
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Word cloud generation failed: {str(e)}"
        )


@app.post("/generate_trend_graph")
def generate_trend_graph(payload: dict):

    try:
        sentiment_data = payload.get("sentiment_data")
        if not sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        df = pd.DataFrame(sentiment_data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        df["sentiment"] = df["sentiment"].astype(int)

        monthly_counts = (
            df
            .resample("M")["sentiment"]
            .value_counts()
            .unstack(fill_value=0)
        )

        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for val in [-1, 0, 1]:
            if val not in monthly_percentages.columns:
                monthly_percentages[val] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))

        color_map = {-1: "red", 0: "gray", 1: "green"}

        for sentiment_val in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_val],
                marker="o",
                linestyle="-",
                label=sentiment_label_map[sentiment_val],
                color=color_map[sentiment_val],
            )

        plt.title("Monthly Sentiment Percentage Over Time")
        plt.xlabel("Month")
        plt.ylabel("Percentage of Comments (%)")
        plt.grid(True)
        plt.xticks(rotation=45)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format="PNG")
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Trend graph generation failed: {str(e)}"
        )