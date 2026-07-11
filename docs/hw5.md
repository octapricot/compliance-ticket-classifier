# HW5 -- CI/CD for Machine Learning

**Goal:** automate delivery of the model to production with a pipeline covering **training** and **deployment**.

**Tool chosen:** GitHub Actions + GitHub Container Registry (GHCR)

**Pipeline:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)
**Published image:** `ghcr.io/octapricot/compliance-ticket-classifier`

---

## 1. Triggers

The task explicitly asks for clearly described triggers. Four are configured:

```yaml
on:
  push:
    branches: [main]          # every merge to main
  pull_request:
    branches: [main]          # validate before merging
  workflow_dispatch:          # manual run from the Actions tab
  schedule:
    - cron: "0 6 * * 1"       # weekly retrain, Mondays 06:00 UTC
```

| Trigger | Purpose |
|---|---|
| `push` | Continuous integration, nothing lands on `main` unbuilt |
| `pull_request` | Catch failures before merge, not after |
| `workflow_dispatch` | On-demand retrain/redeploy without an empty commit |
| `schedule` | The dataset grows as more tickets are labeled; a weekly retrain picks that up automatically |


---

## 2. Stages

```
┌────────┐     ┌──────────────────────┐     ┌────────────────────────────┐
│  lint  │ ──► │        train         │ ──► │          deploy            │
│        │     │                      │     │                            │
│  ruff  │     │  dvc pull  (from S3) │     │  build Docker image        │
│        │     │  train baseline      │     │  start container           │
│        │     │  log metrics > W&B   │     │  wait for /health          │
│        │     │  log model artifact  │     │  smoke test POST /predict  │
│        │     │                      │     │  publish to GHCR           │
└────────┘     └──────────────────────┘     └────────────────────────────┘
```

---

## 3. The `train` job and why it needs S3

```yaml
train:
  needs: lint
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: "3.12"}

    - name: Install dependencies
      run: pip install dvc[s3] pandas scikit-learn wandb python-dotenv joblib pyarrow

    - name: Pull dataset from S3 (DVC)
      env:
        AWS_ACCESS_KEY_ID:     ${{ secrets.AWS_ACCESS_KEY_ID }}
        AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
      run: |
        dvc pull data/labeled/train.parquet data/labeled/test.parquet -v
        ls -la data/labeled/

    - name: Train baseline + log to W&B
      env:
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      run: python src/train/train_baseline.py
```

The training data is not in Git, only content-hash pointers are (see [hw1.md](hw1.md)). So a fresh CI runner has `train.parquet.dvc` (five lines of YAML) and nothing else. `dvc pull` reads the hash, fetches that exact object from S3, and reconstructs the dataset byte-for-byte.

### Which model does CI train?

The **baseline** (TF-IDF + LogReg), not DistilBERT. This is a deliberate constraint, not laziness:

- The baseline trains in circa 30 seconds on CPU. A GitHub-hosted runner has no GPU.
- Fine-tuning DistilBERT on CPU would take hours and time the job out.

DistilBERT is trained in a Colab notebook and promoted to the registry manually. A production setup would dispatch GPU training to a self-hosted runner or a cloud job.

---

## 4. The `deploy` job

```yaml
deploy:
  needs: train
  permissions:
    contents: read
    packages: write          # required for GHCR push
  steps:
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}    # injected automatically

    - name: Build image (load locally for smoke test)
      uses: docker/build-push-action@v6
      with: {context: ., push: false, load: true, tags: compliance-classifier:test}

    - name: Smoke test the service
      env:
        WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
      run: |
        docker run -d --name classifier -p 8000:8000 \
          -e WANDB_API_KEY="$WANDB_API_KEY" compliance-classifier:test
        # poll /health — the container downloads the model from the W&B registry on boot
        for i in $(seq 1 30); do
          curl -sf http://localhost:8000/health > /dev/null && break
          sleep 10
        done
        curl -sf -X POST http://localhost:8000/predict \
          -H "Content-Type: application/json" \
          -d '{"text": "Add user data deletion endpoint for GDPR"}' | tee /tmp/pred.json
        grep -q "label" /tmp/pred.json

    - name: Publish image to GHCR
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}      # :latest and :<short-sha>
```

### Build, then test, then publish -- in that order

The image is built **without pushing**, loaded locally, started, and exercised with a prediction-like request. Only if that passes is it published. 

The smoke test is also what verifies HW3's registry requirement end to end: the container has no baked-in model and no cache, so `/health` only returns 200 once it has genuinely downloaded `compliance-classifier:production` from the W&B registry.

### Tagging

Every image gets two tags:

| Tag | Meaning |
|---|---|
| `latest` | The current `main` |
| `<short-sha>` (e.g. `5af9f15`) | This exact commit |

The SHA tag makes deploys traceable and rollbacks possible. `latest` alone is not a deployment strategy; the same argument as `:latest` vs `:production` in the model registry ([hw2.md](hw2.md)).

```bash
docker pull ghcr.io/octapricot/compliance-ticket-classifier:5af9f15
```

---

## 5. Secrets

| Secret | Used by | Why |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | `train` | `dvc pull` from S3 |
| `AWS_SECRET_ACCESS_KEY` | `train` | `dvc pull` from S3 |
| `WANDB_API_KEY` | `train`, `deploy` | Log metrics; download model from registry in the smoke test |

`GITHUB_TOKEN` is **not** configured manually: GitHub injects it, and `permissions: packages: write` is what authorises the GHCR push.

The AWS credentials belong to a dedicated IAM user (`dvc-user`) whose policy is scoped to exactly one S3 bucket. 

### A debugging note worth recording

The first run of this pipeline failed at `dvc pull` with:

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

The cause: three separate secrets had been created as a single secret containing an `.env`-style blob. GitHub does not parse secret values: one secret is one name mapped to one opaque string. `${{ secrets.AWS_ACCESS_KEY_ID }}` therefore resolved to an **empty string**, silently, with no warning.

The fix was a diagnostic step that prints the *length* of each secret without revealing its value:

```yaml
- name: Check secrets are present
  env:
    AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
  run: |
    echo "Key ID length: ${#AWS_ACCESS_KEY_ID}"     # 20 = valid, 0 = missing
    [ -n "$AWS_ACCESS_KEY_ID" ] || exit 1
```

An AWS access key ID is exactly 20 characters and the secret exactly 40. Anything else is wrong. This step is retained in the pipeline.

---

## 6. What is not automated

- **DistilBERT is not retrained in CI** (no GPU on the runner; see above).
- **Model promotion to `:production` is manual.** This is arguably correct: promoting a model to production should be a human decision informed by its metrics, not an automatic consequence of a push. But it does mean the scheduled weekly retrain produces a new *artifact* without changing what is *served*.
- **There is no deployment to a live environment.** "Deploy" here means publishing a versioned, immutable, pullable image to a registry. The `legal_rag` project ([HW6](https://github.com/octapricot/legal_rag)) goes further, with Terraform-provisioned ECS Fargate and an `aws ecs update-service --force-new-deployment` step; that pattern would be the natural extension here.
- **Drift detection is not wired into the pipeline**, though `src/monitor/drift.py` exits non-zero on drift specifically so that it could be (see [hw4.md](hw4.md)).
- **`ruff` runs with `|| true`**: lint issues are reported but do not fail the build. 
