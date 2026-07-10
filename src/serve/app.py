"""
FastAPI service for the compliance ticket classifier.

Endpoints:
  GET  /health   - liveness check
  POST /predict  - classify one ticket's text

The model is loaded from the W&B registry at startup (see model.py).
Run:  uvicorn src.serve.app:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from src.serve.model import load_model, predict


# Load the model once when the service starts up.
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Compliance Ticket Classifier", lifespan=lifespan)


class TicketIn(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def classify(ticket: TicketIn):
    result = predict(ticket.text)
    return result
