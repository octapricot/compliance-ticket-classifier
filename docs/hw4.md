# HW4 -- Monitoring and Observability

**Goal:** collect metrics from the running inference service, build a dashboard, and (bonus) detect data drift.

**Tools chosen:** Prometheus + Grafana (metrics) + Evidently (drift) + `prometheus-fastapi-instrumentator` (instrumentation)

---

## 1. What is being monitored, and why

Two different questions, answered by two different systems:

| Question | Tool | Failure it catches |
|---|---|---|
| *Is the service healthy?* | Prometheus + Grafana | Crashes, latency spikes, traffic collapse |
| *Is the model still correct?* | Evidently | Silent accuracy decay from distribution shift |

A model whose input distribution has drifted away from its training data keeps returning HTTP 200 with confident-looking probabilities that are quietly wrong. For a compliance screening tool, that is the dangerous failure mode: **the team believes they are covered when they are not.**

---

## 2. Instrumenting the service

Two layers of metrics in `src/serve/app.py`:

```python
# Layer 1: standard HTTP metrics (request count, duration histogram, status codes)
Instrumentator().instrument(app).expose(app)     # >> GET /metrics

# Layer 2: a domain-specific counter
PREDICTIONS = Counter(
    "compliance_predictions_total",
    "Total predictions made, labeled by result",
    ["label"],
)

@app.post("/predict")
def classify(ticket: TicketIn):
    result = predict(ticket.text)
    PREDICTIONS.labels(label=result["label"]).inc()
    log_prediction(ticket.text, result)
    return result
```

The `compliance_predictions_total` counter is labelled by outcome, which means the **ratio** of `relevant` to `not_relevant` predictions is observable over time. That ratio is itself an early warning signal: if the model suddenly stops flagging anything as relevant, something has changed: either in the traffic or in the model; and it will show up here before anyone notices missed tickets.

A Prometheus `Counter` only ever increases; rates are computed at query time.

---

## 3. Prometheus + Grafana

```bash
# The inference service must be running first
python -m uvicorn src.serve.app:app --port 8000

docker-compose up -d
# Prometheus >> http://localhost:9090
# Grafana    >> http://localhost:3000   (admin / admin)
```

`monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 5s

scrape_configs:
  - job_name: "compliance-classifier"
    metrics_path: "/metrics"
    static_configs:
      - targets: ["host.docker.internal:8000"]
```

### Dashboard

Import `monitoring/grafana_dashboard.json`. Three panels:

| Panel | PromQL | Answers |
|---|---|---|
| Request rate (req/s) | `sum(rate(http_requests_total[1m]))` | Predictions per minute: is anyone using this? |
| p95 latency | `histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[1m])) by (le))` | Is it fast enough to run on every ticket? |
| Predictions by label | `sum by (label) (compliance_predictions_total)` | Is the model's output distribution stable? |

**p95, not mean.** The mean hides the tail. If 5% of requests take four seconds, the mean stays comfortable and those 5% of users are the ones who complain. p95 says: 95% of requests are at least this fast.

---

## 4. Drift detection with Evidently

### Prediction logging

Drift detection needs *current* data, so the service records what it actually sees:

```python
PREDICTION_LOG = Path("data/monitoring/predictions.jsonl")

def log_prediction(text: str, result: dict):
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
        print(f"[warn] failed to log prediction: {e}")     # never break the request
```

The `try/except` is not defensive padding. **Monitoring must never be able to take down the thing it monitors.** If the disk fills, the service stops logging and keeps serving.

One line of JSON per prediction (JSONL): appendable, trivially streamable back into pandas.

### The comparison

Raw text cannot be compared statistically, so `src/monitor/drift.py` compares **derived features**:

| Feature | Why |
|---|---|
| `text_length` | Characters per ticket: captures a shift in how teams write |
| `word_count` | Words per ticket: same, less sensitive to formatting |

**Reference** = `train.parquet` (what the model was trained on).
**Current** = the logged predictions (what it is actually seeing).

Evidently runs a **Kolmogorov–Smirnov test** per column. K-S tests the null hypothesis *"these two samples come from the same distribution."* A p-value below 0.05 rejects that hypothesis: the distributions differ.

```python
definition = DataDefinition(numerical_columns=["text_length", "word_count"])
report = Report(metrics=[DataDriftPreset()])
result = report.run(
    reference_data=Dataset.from_pandas(reference, data_definition=definition),
    current_data=Dataset.from_pandas(current, data_definition=definition),
)
result.save_html("data/monitoring/drift_report.html")
```

### Running it

```bash
python src/monitor/drift.py              # against real logged predictions
python src/monitor/drift.py --simulate   # against synthetically drifted data
```

```
Reference (training data): 491 rows
Current (SIMULATED drift): 200 rows

Drift report written to data/monitoring/drift_report.html

=== Drift summary ===
  Drifted columns: 2 (100% of columns)
  text_length     p=8.85e-39  -> DRIFTED
  word_count      p=3.81e-40  -> DRIFTED

/!\  DATASET DRIFT DETECTED - the model may need retraining.
```

**Exit code 1 when drift is detected.** It can gate a CI job or fire an alert. 

The dataset-level rule: if more than **50%** of columns drift, the dataset as a whole is flagged (`Share of Drifted Columns = 1.0` against a threshold of `0.5`). That is the number an alert would be wired to.

### Why `--simulate` exists

Real drift takes weeks or months to appear. That does not fit in a demonstration. The `--simulate` flag deliberately shifts the current distribution: modelling a team that has started writing much longer, more detailed tickets:

```python
drifted["text_length"] = (drifted["text_length"] * 2.5 + 300).astype(int)
drifted["word_count"]  = (drifted["word_count"]  * 2.2 +  40).astype(int)
```

Evidently detects it, as it should. 

---

## 5. An important caveat about false positives

Running `drift.py` against a small handful of test requests **will report drift**, and it will be *correct* to do so. Twenty repetitions of two hand-written sentences genuinely is a different distribution from 491 varied real tickets.

This is the classic false-positive mode of drift detection, and it matters:

- Drift detection on a small sample produces false alarms.
- False alarms train people to ignore the alerts.
- Ignored alerts are worse than no alerts, because they create the *impression* of monitoring.

A production configuration would set a **minimum sample size** (say, 200 predictions) before the check fires at all, and run over a **rolling window** rather than the entire log. The current implementation does neither, and that is a known limitation rather than an oversight.

---

## 6. Limitations

- **No minimum sample size / rolling window.** As above, the script will happily report drift on 20 rows.
- **No alerting.** The exit code makes alerting *possible*, but nothing is wired to Alertmanager or a Slack webhook.
- **Drift detection is not automated.** The natural next step is to add it as a scheduled CI job alongside the existing weekly retrain trigger (see [hw5.md](hw5.md)).
- **No concept-drift detection.** Data drift (inputs changed) is monitored; *concept* drift (the relationship between inputs and correct labels changed -- e.g. a new regulation makes previously-irrelevant tickets relevant) is not, and cannot be without ground-truth labels on live traffic.
- **Prediction logs contain raw ticket text.** Real tickets can contain personal data. The log is gitignored, but a production deployment would need a retention policy, access control, and a documented lawful basis. 
