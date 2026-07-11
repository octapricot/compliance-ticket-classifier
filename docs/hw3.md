# HW3 -- Model Inference

**Goal:** deploy the model as a service, loading it from the version registry configured in HW2.

**Tools chosen:** FastAPI + Docker + W&B Model Registry (model source)

---

## 1. The key requirement

The model must be **pulled from the registry**, not committed to the repo or baked into the image. 

### How it works

`src/serve/model.py`, at startup:

```python
WANDB_ARTIFACT = "k-dubas-set-university-org/wandb-registry-model/compliance-classifier:production"

def _download_model():
    api = wandb.Api()
    artifact = api.artifact(WANDB_ARTIFACT, type="model")
    artifact.download(root=str(LOCAL_MODEL_DIR))
    return LOCAL_MODEL_DIR
```

Observe what is being requested: **`compliance-classifier:production`** -- a registry collection and a deliberate alias. Not `distilbert-compliance:latest`, which would mean "whatever ran most recently," including a failed experiment. See [hw2.md](hw2.md) for why that distinction matters.

Startup log:

```
Downloading model from W&B registry...
wandb: Downloading large artifact 'compliance-classifier:production', 256.11MB. 5 files...
wandb:   5 of 5 files downloaded.
Model downloaded to models/serving_model
Model loaded and ready.
```

The model is cached in `models/serving_model/` after the first download, so restarts are fast. Deleting that directory forces a fresh pull, which is how the registry path is tested.

---

## 2. The service

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()     # download + load once, at startup (not per request)
    yield

app = FastAPI(title="Compliance Ticket Classifier", lifespan=lifespan)
```

Loading in the `lifespan` hook rather than on first request means the container is either *ready* or *not ready*, never "up but about to spend 30 seconds downloading a model mid-request." That is what makes the `/health` check meaningful, and it is what the CI smoke test relies on.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness — returns `{"status": "ok"}` once the model is loaded |
| `/predict` | POST | Classify one ticket |
| `/metrics` | GET | Prometheus metrics (see [hw4.md](hw4.md)) |

### Prediction

```python
def predict(text: str):
    inputs = _tokenizer(text, truncation=True, max_length=256,
                        padding="max_length", return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    pred_idx = int(torch.argmax(probs))
    return {
        "label": LABELS[pred_idx],
        "confidence": round(float(probs[pred_idx]), 4),
        "probabilities": {
            "not_relevant": round(float(probs[0]), 4),
            "relevant":     round(float(probs[1]), 4),
        },
    }
```

The full probability distribution is returned, not just the argmax. This is deliberate: given the recall problem documented in [hw2.md](hw2.md), a downstream consumer may well want to apply its own threshold; flagging anything above `prob_relevant > 0.3` for human review, for instance, rather than accepting the model's 0.5 default. Returning only a label would foreclose that.

---

## 3. Running it

### From source

```bash
python -m uvicorn src.serve.app:app --port 8000
```

### From the published image

```bash
docker pull ghcr.io/octapricot/compliance-ticket-classifier:latest

docker run -p 8000:8000 \
  -e WANDB_API_KEY=<your-key> \
  ghcr.io/octapricot/compliance-ticket-classifier:latest
```

The image is published by CI on every push to `main` (see [hw5.md](hw5.md)), tagged with both `latest` and the commit SHA. It contains the *application*, not the model; the model is fetched from the registry at container start. `WANDB_API_KEY` is required for that fetch.

### Docker image notes

```dockerfile
FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
```

CPU-only torch is installed explicitly first. The default `pip install torch` pulls circa 5 GB of CUDA libraries that are useless for CPU inference and would make the image absurd.

---

## 4. Example requests

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Add user data deletion endpoint for GDPR compliance"}'
```

```json
{
  "label": "relevant",
  "confidence": 0.8398,
  "probabilities": {"not_relevant": 0.1602, "relevant": 0.8398}
}
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Fix memory leak in the etcd client connection pool"}'
```

```json
{
  "label": "not_relevant",
  "confidence": 0.9072,
  "probabilities": {"not_relevant": 0.9072, "relevant": 0.0928}
}
```

Health:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## 5. Verified from CI

The registry pull is not only tested locally. The CI pipeline's `deploy` job:

1. builds the image,
2. starts the container,
3. polls `/health` until the container has downloaded the model **from the registry** and is ready,
4. sends a real `POST /predict` and checks the response,
5. only then publishes the image to GHCR.

That means the "pulls from the registry" claim is verified on every push, by a machine with no local cache and no relationship to the development laptop. If the registry path breaks, the pipeline goes red and no image ships.

---

## 6. Limitations

- **No batching.** One ticket per request. For scanning a full backlog, a `/predict_batch` endpoint would be significantly more efficient.
- **No authentication.** The service is open. A production deployment would need at minimum an API key, since the endpoint accepts arbitrary text and logs it.
- **Cold start is slow** (circa 30-60s) because the model is downloaded on first boot. Baking the model into the image would fix this but would defeat the registry requirement: the right answer is a persistent volume cache, as used in the [legal_rag](https://github.com/octapricot/legal_rag) EFS setup.
- **Every request is logged to disk** including raw ticket text (see [hw4.md](hw4.md)). This is necessary for drift detection but has data-protection implications that a real deployment must address.
