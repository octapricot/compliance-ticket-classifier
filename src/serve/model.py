"""
Loads the compliance classifier from the W&B model registry and runs inference.

The model is downloaded from W&B once at startup (not hardcoded in the repo),
satisfying the 'pull from registry' requirement.
"""
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()  # loads WANDB_API_KEY

# W&B artifact reference: <entity>/<project>/<artifact_name>:<version>
# Pull from the W&B MODEL REGISTRY, not from a raw project artifact.
WANDB_ARTIFACT = "k-dubas-set-university-org/wandb-registry-model/compliance-classifier:production"
LOCAL_MODEL_DIR = Path("models/serving_model")
LABELS = {0: "not_relevant", 1: "relevant"}
MAX_LEN = 256

# Module-level cache so we load the model only once.
_tokenizer = None
_model = None


def _download_model():
    """Download the model artifact from W&B into LOCAL_MODEL_DIR."""
    import wandb
    if LOCAL_MODEL_DIR.exists() and any(LOCAL_MODEL_DIR.iterdir()):
        print(f"Model already present at {LOCAL_MODEL_DIR}, skipping download.")
        return LOCAL_MODEL_DIR

    print("Downloading model from W&B registry...")
    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT, type="model")
    artifact.download(root=str(LOCAL_MODEL_DIR))
    print(f"Model downloaded to {LOCAL_MODEL_DIR}")
    return LOCAL_MODEL_DIR


def load_model():
    """Load tokenizer + model into memory (once)."""
    global _tokenizer, _model
    if _model is not None:
        return
    model_dir = _download_model()
    _tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    _model.eval()  # inference mode
    print("Model loaded and ready.")


def predict(text: str):
    """Return {label, confidence, probabilities} for one ticket text."""
    if _model is None:
        load_model()

    inputs = _tokenizer(
        text, truncation=True, max_length=MAX_LEN,
        padding="max_length", return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = int(torch.argmax(probs))
    return {
        "label": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            "not_relevant": round(float(probs[0]), 4),
            "relevant": round(float(probs[1]), 4),
        },
    }
