# Compliance Ticket Classifier

A machine-learning system that flags (or, at least, is intended to :) software-development tickets (Jira / GitHub issues) as **compliance-relevant** (GDPR only for now) or not, wrapped in a full MLOps lifecycle: data labeling, dataset versioning, experiment tracking, a model registry, inference serving, monitoring, and CI/CD.

---

## The problem

Development teams routinely create tickets that trigger data-protection obligations **without using any privacy vocabulary**. A ticket that reads *"add log storage microservice"* contains no mention of GDPR, personal data, or privacy; yet it may create a system subject to data-protection-by-design requirements under Art. 25 GDPR.

Keyword search cannot catch this. A Data Protection Officer reviewing every ticket by hand does not scale. This project trains a classifier to surface such tickets automatically, using a labeling schema derived from actual DPO practice.

**Why the cost of errors is asymmetric:** a false positive costs a DPO thirty seconds of review. A false negative is an unflagged legal obligation that nobody knows exists. This drives every modelling decision below: recall matters more than accuracy.

---

## Architecture

```
   Raw issues (HuggingFace)
            │
            ▼
   ┌─────────────────┐
   │  Pre-filter     │  keyword + exposure heuristics
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Label Studio   │  321 tickets labeled by DPO criteria
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  DVC  >>  S3    │  dataset versioned (data-v1, data-v2)
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  Training       │  TF-IDF baseline + DistilBERT
   │  W&B tracking   │  metrics, hyperparameters, artifacts
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  W&B Registry   │  model promoted with :production alias
   └────────┬────────┘
            ▼
   ┌─────────────────┐
   │  FastAPI        │  pulls :production from registry at startup
   │  /predict       │
   └────────┬────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌──────────┐  ┌──────────────┐
│Prometheus│  │  Evidently   │
│+ Grafana │  │  drift check │
└──────────┘  └──────────────┘

   GitHub Actions ties it together:
   lint >> train (dvc pull from S3) >> build >> smoke test >> publish to GHCR
```

---

## Homework index

| # | Topic | Tools | Details |
|---|---|---|---|
| 1 | Data labeling & versioning | Label Studio, DVC, AWS S3 | [docs/hw1.md](docs/hw1.md) |
| 2 | Training & experiment tracking | W&B, scikit-learn, HuggingFace | [docs/hw2.md](docs/hw2.md) |
| 3 | Model inference | FastAPI, Docker, W&B Registry | [docs/hw3.md](docs/hw3.md) |
| 4 | Monitoring & observability | Prometheus, Grafana, Evidently | [docs/hw4.md](docs/hw4.md) |
| 5 | CI/CD | GitHub Actions, GHCR | [docs/hw5.md](docs/hw5.md) |
| 6 | RAG system | **Separate repo >>** [octapricot/legal_rag](https://github.com/octapricot/legal_rag) | see below |

**Deep Learning capstone.** This same model is also the subject of a Deep Learning project. The modelling work itself (data strategy, synthetic augmentation, ablation study, interpretability, threshold tuning) is documented separately in **[docs/capstone.md](docs/capstone.md)**. This README covers the MLOps lifecycle *around* that model.

---

## Setup

```bash
git clone https://github.com/octapricot/compliance-ticket-classifier
cd compliance-ticket-classifier

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # add WANDB_API_KEY
```

**To retrieve the dataset** (it is not in Git, only DVC pointers are):

```bash
# Configure your own S3 credentials for the DVC remote
dvc remote modify --local storage access_key_id     <AWS_ACCESS_KEY_ID>
dvc remote modify --local storage secret_access_key <AWS_SECRET_ACCESS_KEY>

dvc pull
```

---

## HW1 -- Data labeling and dataset versioning

**Labeling tool:** [Label Studio](https://labelstud.io) (self-hosted)
**Storage:** AWS S3 (`eu-central-1`)
**Versioning:** DVC

### What was done

Raw Jira/GitHub issues were pulled from the HuggingFace dataset `hankzhwang/issues`, which has **no compliance labels**, and creating them was the core of the work. Tickets were pre-filtered by keyword and data-exposure heuristics, exported to Label Studio, and labeled by hand against DPO criteria.

**321 tickets labeled** across two Label Studio projects (121 + 200), both 100% complete. 

### Run the labeling

```bash
# 1. Pull and pre-filter raw issues
python src/data/pull_raw.py
python src/data/prefilter.py

# 2. Export tasks for Label Studio
python src/data/make_label_studio_tasks.py
# >> data/candidates/label_studio_tasks.json

# 3. Start Label Studio and import that JSON
label-studio start        # >> http://localhost:8080

# 4. After labeling, export back out and assemble the dataset
python src/data/extract_gold.py
python src/data/assemble_dataset.py
```

### How versioning works

Large files never enter Git. DVC stores the **content hash** in a small `.dvc` pointer file (which *is* committed) and the actual bytes in S3, named by that hash:

```yaml
# data/labeled/gold.parquet.dvc
outs:
- md5: 0f5891b999ef5aa1d57046380dabdedf
  size: 273467
  path: gold.parquet
```

Two dataset versions exist, tagged in Git:

| Tag | md5 | Size | Content |
|---|---|---|---|
| `data-v1` | `0364b946…` | 105 KB | First labeling round (121 tickets) |
| `data-v2` | `0f5891b9…` | 273 KB | Expanded round (321 tickets) |

**Same filename, different content.** Switching between them:

```bash
git checkout data-v1 && dvc checkout
python -c "import pandas as pd; print(len(pd.read_parquet('data/labeled/gold.parquet')))"

git checkout data-v2 && dvc checkout
python -c "import pandas as pd; print(len(pd.read_parquet('data/labeled/gold.parquet')))"
```


### Data lineage

```
hankzhwang/issues  >  pull_raw.py  >  prefilter.py  >  Label Studio
                                                             │
                                        extract_gold.py  ◄───┘
                                                │
                              assemble_dataset.py  >  train / test / gold  >  DVC > S3
```

**>> [Full details: docs/hw1.md](docs/hw1.md)**

---

## HW2 -- Training and experiment tracking

**Tracker:** [Weights & Biases](https://wandb.ai/k-dubas-set-university/compliance-ticket-classifier)
**Registry:** [W&B Model Registry](https://wandb.ai/orgs/k-dubas-set-university-org/registry/model)

### What was done

Two models were trained on the same versioned dataset, so their numbers are directly comparable:

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.76 | 0.80 | 0.29 | 0.43 |
| **DistilBERT (fine-tuned)** | — | **0.83** | **0.46** | **0.59** |

Every run logs metrics, hyperparameters, a confusion matrix, and the trained model as a **versioned artifact**.

### Run training

```bash
# Baseline (CPU, circa 30 seconds)
python src/train/train_baseline.py

# Transformer (GPU): notebooks/train_distilbert.ipynb, run in Colab
```

Results appear at the [W&B project](https://wandb.ai/k-dubas-set-university/compliance-ticket-classifier) under **Runs** and **Artifacts**.

### Artifacts vs. Registry -- and why the distinction matters

`wandb.log_artifact()` stores *every* model ever trained (`v0`, `v1`, …) (version tracking, not a release process).

The **Model Registry** is separate: a curated collection, where a specific version is deliberately **promoted**. The inference service pulls the `:production` alias -- **not** `:latest`. 

```bash
python src/train/register_model.py    # promotes an artifact >> registry, alias :production
```

**>> [Full details: docs/hw2.md](docs/hw2.md)**

---

## HW3 -- Model inference

**Serving:** FastAPI + Docker
**Model source:** W&B Model Registry (`compliance-classifier:production`)

### Run the service

```bash
# From source
python -m uvicorn src.serve.app:app --port 8000

# Or from the published image
docker pull ghcr.io/octapricot/compliance-ticket-classifier:latest
docker run -p 8000:8000 -e WANDB_API_KEY=<key> \
  ghcr.io/octapricot/compliance-ticket-classifier:latest
```

At startup the service **downloads the model from the registry**, it is not baked into the repository or the image.

### Send a request

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

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /predict` | Classify one ticket |
| `GET /metrics` | Prometheus metrics |

**>> [Full details: docs/hw3.md](docs/hw3.md)**

---

## HW4 -- Monitoring and observability

**Metrics:** Prometheus + Grafana
**Drift:** Evidently

### Start the monitoring stack

```bash
# 1. Start the inference service (must be running first)
python -m uvicorn src.serve.app:app --port 8000

# 2. Start Prometheus + Grafana
docker-compose up -d

# Grafana  >> http://localhost:3000   (admin / admin)
# Prometheus >> http://localhost:9090
```

Import `monitoring/grafana_dashboard.json` into Grafana. The dashboard shows:

| Panel | PromQL |
|---|---|
| Request rate (predictions/min) | `sum(rate(http_requests_total[1m]))` |
| p95 latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))` |
| Predictions by label | `sum by (label) (compliance_predictions_total)` |

### Drift detection

The service logs every prediction to `data/monitoring/predictions.jsonl`. `src/monitor/drift.py` compares that live traffic against the training set:

```bash
python src/monitor/drift.py              # real logged predictions
python src/monitor/drift.py --simulate   # inject synthetic drift, for demonstration
```

```
=== Drift summary ===
  Drifted columns: 2 (100% of columns)
  text_length     p=8.85e-39  -> DRIFTED
  word_count      p=3.81e-40  -> DRIFTED

/!\  DATASET DRIFT DETECTED - the model may need retraining.
```

It writes an interactive HTML report to `data/monitoring/drift_report.html` and **exits non-zero when drift is detected**.

**Why this matters here:** the model's weights are frozen at training time. If production tickets drift away from the training distribution, the model quietly gets worse: no error, no alert. For a compliance screening tool, that is the dangerous failure mode: the team believes they are covered when they are not.

**>> [Full details: docs/hw4.md](docs/hw4.md)**

---

## HW5 -- CI/CD

**Pipeline:** [`.github/workflows/ci.yml`](.github/workflows/ci.yml)
**Registry:** [GitHub Container Registry](https://github.com/octapricot/compliance-ticket-classifier/pkgs/container/compliance-ticket-classifier)

### Triggers

| Trigger | When |
|---|---|
| `push` to `main` | Every merge |
| `pull_request` | Validate before merging |
| `workflow_dispatch` | Manual run from the Actions tab |
| `schedule` | Weekly retrain: Mondays 06:00 UTC |

### Stages

```
lint  ──►  train  ──►  deploy
 │          │            │
 │          │            ├─ build Docker image
 │          │            ├─ start container, wait for /health
 │          │            ├─ smoke test POST /predict
 │          │            └─ publish to GHCR (tagged :latest and :<sha>)
 │          │
 │          ├─ dvc pull  (fetch versioned dataset from S3)
 │          ├─ train baseline
 │          └─ log metrics + model artifact to W&B
 │
 └─ ruff
```

Jobs are chained with `needs:`, so **an image is never published from a build whose training failed**.

### Pull the published image

```bash
docker pull ghcr.io/octapricot/compliance-ticket-classifier:latest
```

### Required GitHub secrets

| Secret | Used by |
|---|---|
| `AWS_ACCESS_KEY_ID` | `dvc pull` from S3 |
| `AWS_SECRET_ACCESS_KEY` | `dvc pull` from S3 |
| `WANDB_API_KEY` | Experiment logging + registry download |

`GITHUB_TOKEN` is injected automatically for the GHCR push.

**>> [Full details: docs/hw5.md](docs/hw5.md)**

---

## HW6 -- RAG system

**Separate repository: [octapricot/legal_rag](https://github.com/octapricot/legal_rag)**

A GDPR research assistant that answers natural-language legal questions by retrieving exact source passages and generating structured briefs with **verbatim citations**.

| Component | Choice |
|---|---|
| Corpus | GDPR (99 articles + 173 recitals) + 43 EDPB guidelines >> **4,633 chunks** |
| Chunking | Structure-aware, split at article/section boundaries |
| Retrieval | **Hybrid**: BM25 + dense (`bge-large-en-v1.5`), fused with Reciprocal Rank Fusion, then cross-encoder reranked |
| Vector store | ChromaDB |
| LLM | Mistral 7B via Ollama (dev) / Claude Haiku (prod) |
| Interface | FastAPI + web UI |
| Infrastructure | Terraform >> ECS Fargate, ALB, EFS, ECR |

Every claim in a generated brief carries a citation key *and* the verbatim source text, so a DPO can verify it without leaving the document. The output schema structurally prevents the model from inventing a source.

See the [legal_rag README](https://github.com/octapricot/legal_rag) for architecture and setup.

---

## Limitations and caveats

**Recall is the weak point.** The baseline catches only 29% of compliance-relevant tickets; DistilBERT improves this to 46%. Given that a missed ticket is an unflagged legal obligation, this is the number that matters most, and it is not yet good enough for unsupervised production use. The system is a screening aid for a human DPO, not a replacement. Improving recall (threshold tuning, more labeled data, class-weighted loss) is the highest-value next step.

**Small labeled dataset.** 321 hand-labeled examples, augmented with synthetic positives to address class imbalance. The [ablation study](docs/capstone.md) quantifies what the synthetic data contributed.

**Drift detection needs a representative sample.** Running `drift.py` against a handful of test requests will report drift; correctly, because a handful of identical requests genuinely is a different distribution. In production this should run over a rolling window of hundreds of predictions with a minimum sample size, or it will produce false alarms, and false alarms get monitoring switched off.

**Prediction logs contain raw ticket text.** `data/monitoring/predictions.jsonl` stores the full text of every ticket sent to the service. Real Jira/GitHub issues can contain personal data (usernames, emails in stack traces). The log is gitignored, but a production deployment would need a retention policy, access controls, and a lawful basis; the classifier is a data-protection tool that itself processes personal data.

**Single annotator.** All labels come from one person. No inter-annotator agreement was measured, so the labeling schema's consistency is unverified.

---

## Project structure

```
data/              raw, candidate, and labeled datasets (DVC-tracked, not in Git)
  ├── labeled/     gold.parquet, train/test splits, versioned via DVC >> S3
  └── monitoring/  prediction logs + drift reports (gitignored, runtime only)
src/
  ├── data/        pull, pre-filter, export-for-labeling, assemble
  ├── train/       baseline training, model registry promotion
  ├── serve/       FastAPI inference service
  └── monitor/     Evidently drift detection
models/            trained models (registry-tracked, not in Git)
monitoring/        Prometheus config + Grafana dashboard
notebooks/         exploration, DistilBERT training
.github/workflows/ CI/CD pipeline
docs/              per-homework write-ups + Deep Learning capstone report
```
