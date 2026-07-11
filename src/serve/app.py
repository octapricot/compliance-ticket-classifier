"""
FastAPI service for the compliance ticket classifier.

Endpoints:
  GET  /health   - liveness check
  POST /predict  - classify one ticket's text
  GET  /metrics  - Prometheus metrics (request counts, latency, prediction counts)

The model is loaded from the W&B registry at startup (see model.py).
Run:  python -m uvicorn src.serve.app:app --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator

from src.serve.model import load_model, predict

import json
from datetime import datetime, timezone
from pathlib import Path

# Custom metric: count predictions by their resulting label.
# A Counter only goes up; Prometheus computes rates from it over time.
PREDICTIONS = Counter(
    "compliance_predictions_total",
    "Total predictions made, labeled by result",
    ["label"],
)

# Every prediction is appended here as one JSON object per line (JSONL).
# This is the "current" data that drift detection compares against the
# training set. Without it, we could not tell whether live traffic has
# drifted away from what the model was trained on.
PREDICTION_LOG = Path("data/monitoring/predictions.jsonl")

def log_prediction(text: str, result: dict):
    """Append one prediction to the log. Never breaks the request."""
    try:
        PREDICTION_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "label": result["label"],
            "confidence": result["confidence"],
            "prob_relevant": result["probabilities"]["relevant"],
        }
        with PREDICTION_LOG.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        # Logging must never take down the service.
        print(f"[warn] failed to log prediction: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Compliance Ticket Classifier", lifespan=lifespan)

# Auto-instrument standard HTTP metrics (request count, latency, etc.)
# and expose them at /metrics.
Instrumentator().instrument(app).expose(app)


class TicketIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def classify(ticket: TicketIn):
    result = predict(ticket.text)
    PREDICTIONS.labels(label=result["label"]).inc()
    log_prediction(ticket.text, result)
    return result
