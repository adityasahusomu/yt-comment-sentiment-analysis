# scripts/test_fastapi.py
import os, sys
from fastapi.testclient import TestClient

# ensure we can import FastAPI/app.py
sys.path.append("FastAPI")

# tell the app to skip heavy model loads in CI
os.environ["SKIP_MODEL_LOADING"] = "1"

from app import app  # after env var is set

client = TestClient(app)

def test_predict_endpoint():
    data = {
        "comments": ["This is a great product!", "Not worth the money.", "It's okay."]
    }
    response = client.post("/predict", json=data)  # or requests.post if you're starting a server
    assert response.status_code == 200

    payload = response.json()
    # Top-level keys
    assert isinstance(payload, dict)
    assert "predictions" in payload
    assert "sentiment_counts" in payload

    # predictions list
    preds = payload["predictions"]
    assert isinstance(preds, list)
    assert len(preds) == 3
    for item in preds:
        assert "comment" in item and "sentiment" in item

    # counts object
    counts = payload["sentiment_counts"]
    assert set(counts.keys()) == {"1", "0", "-1"}
    # integers as values
    assert all(isinstance(v, int) for v in counts.values())

def test_predict_with_timestamps_endpoint():
    data = {
        "comments": [
            {"text": "This is fantastic!", "timestamp": "2024-10-25 10:00:00"},
            {"text": "Could be better.", "timestamp": "2024-10-26 14:00:00"}
        ]
    }
    res = client.post("/predict_with_timestamps", json=data)
    assert res.status_code == 200
    assert all("sentiment" in item for item in res.json())

def test_generate_chart_endpoint():
    data = {"sentiment_counts": {"1": 5, "0": 3, "-1": 2}}
    res = client.post("/generate_chart", json=data)
    assert res.status_code == 200
    assert res.headers.get("content-type") == "image/png"

def test_generate_wordcloud_endpoint():
    data = {"comments": ["Love this!", "Not so great.", "Absolutely amazing!", "Horrible experience."]}
    res = client.post("/generate_wordcloud", json=data)
    assert res.status_code == 200
    assert res.headers.get("content-type") == "image/png"

def test_generate_trend_graph_endpoint():
    data = {"sentiment_data": [
        {"timestamp": "2024-10-01", "sentiment": 1},
        {"timestamp": "2024-10-02", "sentiment": 0},
        {"timestamp": "2024-10-03", "sentiment": -1}
    ]}
    res = client.post("/generate_trend_graph", json=data)
    assert res.status_code == 200
    assert res.headers.get("content-type") == "image/png"