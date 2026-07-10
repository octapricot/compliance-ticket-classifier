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


# Custom metric: count predictions by their resulting label.
# A Counter only goes up; Prometheus computes rates from it over time.
PREDICTIONS = Counter(
    "compliance_predictions_total",
    "Total predictions made, labeled by result",
    ["label"],
)


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
    # Increment the counter for whichever label was predicted.
    PREDICTIONS.labels(label=result["label"]).inc()
    return result
