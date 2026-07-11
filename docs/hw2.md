# HW2 -- Model Training and Experiment Tracking

**Goal:** train models, track experiments, and store versioned models in a registry.

**Tools chosen:** scikit-learn + HuggingFace Transformers (training) · Weights & Biases (tracking + registry)

**Project:** https://wandb.ai/k-dubas-set-university/compliance-ticket-classifier
**Registry:** https://wandb.ai/orgs/k-dubas-set-university-org/registry/model

---

## 1. Two models, one dataset

Both models train on the same DVC-versioned dataset from HW1, so their numbers are directly comparable, which is the entire reason for versioning the data in the first place.

### Baseline: TF-IDF + Logistic Regression

```bash
python src/train/train_baseline.py    # CPU, circa 30 seconds
```

The baseline exists for three reasons:
1. It gives a number the transformer must beat.
2. It validates the data pipeline end to end.
3. It is **interpretable**: you can read the learned coefficients directly.

```python
Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                              min_df=2, sublinear_tf=True)),
    ("clf",   LogisticRegression(max_iter=1000, class_weight="balanced")),
])
```

`class_weight="balanced"` compensates for the 2:1 class imbalance by weighting the minority class more heavily in the loss.

### Transformer — fine-tuned DistilBERT

`notebooks/train_distilbert.ipynb`, run on Colab GPU. DistilBERT was chosen over BERT for a practical reason: it is circa 40% smaller and circa 60% faster at inference, and inference latency is a real constraint for a service intended to run on every ticket a team creates.

---

## 2. Results

Evaluated on the **real held-out test set only** (no synthetic examples in the test split: a synthetic test set would measure the ability to model synthetic data, not tickets).

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.76 | 0.80 | 0.29 | 0.43 |
| **DistilBERT** | — | **0.83** | **0.46** | **0.59** |

### Reading the confusion matrix

Baseline, on 132 test tickets:

```
              predicted
              not_rel  rel
true not_rel    88      3     << 3 false alarms
true relevant   29     12     << 29 MISSED
```

**The baseline misses 71% of compliance-relevant tickets.** Precision is high (0.80 when it fires, it is usually right) but it barely fires.

For this application that is the worst possible failure mode, and it is worth stating plainly:

- A **false positive** costs a DPO thirty seconds of review.
- A **false negative** is an unflagged data-protection obligation that nobody knows exists.

The costs are wildly asymmetric, and accuracy (0.76) is blind to that asymmetry. **Recall is the metric that matters.** DistilBERT's improvement from 0.29 to 0.46 is the single most meaningful result here, and it is still not good enough for unsupervised production use. The system is a screening aid, not a replacement for review.

### What the baseline learned

The top coefficients are a useful sanity check that the model latched onto something real rather than an artifact:

| >> RELEVANT | >> NOT RELEVANT |
|---|---|
| `email` (+1.59) | `etcdserver` (−0.72) |
| `users` (+1.37) | `node` (−0.69) |
| `user` (+1.34) | `linux` (−0.57) |
| `user_id` (+0.93) | `connection` (−0.55) |
| `account` (+0.91) | `etcd` (−0.51) |
| `audit` (+0.90) | |

Personal-data vocabulary on one side, infrastructure vocabulary on the other. The model learned the right *distinction*; it is simply too conservative about applying it.

---

## 3. Experiment tracking with W&B

Every run logs:

| What | Where |
|---|---|
| Metrics (accuracy, precision, recall, F1) | `wandb.log()` |
| Hyperparameters (max_features, ngram_range, class_weight, train/test sizes, synthetic count) | `wandb.init(config=…)` |
| Confusion matrix | `wandb.plot.confusion_matrix()` |
| Trained model | `wandb.Artifact(type="model")` |

```python
run = wandb.init(
    project="compliance-ticket-classifier",
    name="baseline-tfidf-logreg",
    config={
        "model": "tfidf+logreg",
        "tfidf_max_features": 5000,
        "tfidf_ngram_range": "1-2",
        "class_weight": "balanced",
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_synthetic": int(train_df["is_synthetic"].sum()),
    },
)
```

Logging `train_synthetic` in the config is deliberate: it is the variable the [ablation study](capstone.md) turns on and off, and recording it per-run is what makes the ablation comparable after the fact rather than a story told from memory.

Multiple comparable runs exist in the project: the baseline, the DistilBERT fine-tune, and the ablation variants plus automated runs triggered by CI (see HW5).

---

## 4. Artifacts vs. Model Registry

This distinction took some care to get right, and it is the substance of the "model registry" requirement.

| | **Artifacts** | **Model Registry** |
|---|---|---|
| What | Versioned file storage. Every `log_artifact()` creates `v0`, `v1`, `v2`… | A curated collection where a chosen version is **promoted** |
| Scope | Belongs to a project | Belongs to the **organisation** |
| Meaning | "Here is every model I ever trained" | "Here is the model we serve" |
| Alias | `latest` = whatever ran most recently | `production` = what a human deliberately chose |

### Why `:latest` is dangerous

CI retrains the baseline on **every push to main**. Each run logs an artifact and moves `latest`. If the inference service pulled `baseline:latest`, then:

- a debugging run at 2am,
- a run with a bug in the data loader,
- a model that scored *worse* than the current one,

would each silently become the model in production. Nothing would fail, predictions would just get worse.

The registry breaks that coupling. Training writes artifacts freely; **serving reads only the `production` alias**, and that alias moves only when someone moves it.

### Promoting a model

```bash
python src/train/register_model.py
```

```python
run.link_artifact(
    artifact=artifact,                                    # distilbert-compliance:v0
    target_path="wandb-registry-model/compliance-classifier",
    aliases=["production"],
)
```

The registry entry carries the model's metrics with it (F1 0.59, Recall 0.46, Precision 0.83 are visible in the registry UI), so candidate versions can be compared **on their numbers** before one is promoted. That is what makes promotion a decision rather than a ritual.

---

## 5. Handoff to HW3

The registry is the **contract** between training and serving. HW3's inference service does exactly this at startup:

```python
WANDB_ARTIFACT = "k-dubas-set-university-org/wandb-registry-model/compliance-classifier:production"
artifact = api.artifact(WANDB_ARTIFACT, type="model")
artifact.download(root="models/serving_model")
```

The model is **not** committed to the repository, not baked into the Docker image, and not selected by recency. It is pulled, by alias, from a registry.

---

## 6. Limitations

- **DistilBERT training lives in a notebook**, not a runnable script. It is reproducible (open in Colab, run all) but not automatable, which is why CI retrains only the baseline.
- **No hyperparameter sweep.** Both models use a single hand-chosen configuration. W&B Sweeps would be the natural next step.
- **Threshold left at 0.5.** Given the recall problem, tuning the decision threshold to trade precision for recall is the cheapest available improvement and has not been done in the served model.
- **Small test set** (132 examples). Confidence intervals on these metrics are wide.
