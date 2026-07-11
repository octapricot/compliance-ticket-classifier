# HW1 -- Data Labeling and Dataset Versioning

**Goal:** set up a labeling system and version the resulting dataset for downstream training.

**Tools chosen:** Label Studio (labeling) + AWS S3 (storage) + DVC (versioning)

---

## 1. The core problem: there were no labels

The starting point was the HuggingFace dataset `hankzhwang/issues` -- public Jira and GitHub issues. It contains ticket text, but **no compliance labels**. Nobody has published a dataset of "software tickets that trigger GDPR obligations," because producing one requires domain expertise rather than annotation labour.

Creating those labels *is* the substance of this homework. The labeling schema comes from Data Protection Officer practice: a ticket is `relevant` if implementing it would create, modify, or expose a processing activity involving personal data, regardless of whether the ticket mentions privacy at all.

That last clause is the whole point. Consider:

| Ticket text | Contains privacy keywords? | Compliance-relevant? |
|---|---|---|
| "Add user data deletion endpoint for GDPR" | Yes | Yes (trivially) |
| "Provide two extra free-form fields to store user information" | **No** | **Yes** -- new personal-data fields, purpose limitation and minimisation apply |
| "Support overwriting existing documents in bulk insert" | No | Depends -- could defeat retention/erasure guarantees |
| "Cannot build on armv6l architecture (raspberry pi)" | No | No |

A keyword filter catches row 1 and misses row 2. Row 2 is the case that matters.

---

## 2. Pipeline

```
hankzhwang/issues
      │
      ▼  src/data/pull_raw.py
raw issues
      │
      ▼  src/data/prefilter.py        keyword + data-exposure heuristics
candidate tickets                      (reduces labeling volume; does NOT decide the label)
      │
      ▼  src/data/make_label_studio_tasks.py
label_studio_tasks.json
      │
      ▼  Label Studio (manual annotation)
annotations
      │
      ▼  src/data/extract_gold.py
gold.parquet
      │
      ▼  src/data/assemble_dataset.py  (+ synthetic augmentation, stratified split)
train.parquet / test.parquet
      │
      ▼  dvc add + dvc push
AWS S3 (content-addressed)
```

### On the pre-filter

`prefilter.py` narrows thousands of raw issues to a labelable candidate pool using keyword and data-exposure signals. **It does not assign labels**, it only decides what a human looks at.

This is a deliberate trade-off with a known cost: any compliance-relevant ticket whose text is *entirely* devoid of the pre-filter's signals never reaches the annotator, and therefore never enters the dataset. The model cannot learn to catch what it was never shown. This biases the dataset toward the tickets that are *somewhat* detectable, and is a real limitation of the labeled set, not a solved problem.

The alternative (label a random sample of thousands of issues, where positives are rare) was not feasible by hand.

---

## 3. Labeling in Label Studio

Two projects, both completed:

| Project | Tasks | Complete |
|---|---|---|
| Compliance Ticket Review | 121 | 100% |
| Compliance Ticket Review #2 | 200 | 100% |

**321 tickets labeled.** The task required "a few dozen."

Final label distribution in `gold.parquet` (318 rows after de-duplication):

| Label | Count |
|---|---|
| `not_relevant` | 215 |
| `relevant` | 103 |

Roughly 1:2 -- imbalanced but workable. The imbalance is addressed at training time with synthetic augmentation and class weighting (see [capstone.md](capstone.md)).

Each task carries the ticket text plus supporting fields (`source`, `keyword_hits`, `exposure_hits`, and an `ai_reason` field capturing a draft rationale) to make consistent human judgement faster.

### Reproducing the labeling

```bash
python src/data/pull_raw.py
python src/data/prefilter.py
python src/data/make_label_studio_tasks.py      # >> data/candidates/label_studio_tasks.json

label-studio start                               # >> http://localhost:8080
# Create a project >> Import >> select the JSON above
# Labeling interface: single choice, {relevant, not_relevant}

# After labeling: export from Label Studio, then
python src/data/extract_gold.py
python src/data/assemble_dataset.py
```

---

## 4. Versioning with DVC

### How DVC resolves this

DVC splits the file from its identity:

- The **bytes** go to S3.
- A **pointer** goes to Git.

```yaml
# data/labeled/gold.parquet.dvc  -- this is committed
outs:
- md5: 0f5891b999ef5aa1d57046380dabdedf
  size: 273467
  hash: md5
  path: gold.parquet
```

### Configuration

```bash
dvc remote add -d storage s3://mlops-compliance-classifier-kdubas/dvcstore
dvc remote modify storage region eu-central-1

# Credentials go to .dvc/config.local: gitignored, never committed
dvc remote modify --local storage access_key_id     <KEY_ID>
dvc remote modify --local storage secret_access_key <SECRET>
```

The split matters:

| File | Committed? | Contains |
|---|---|---|
| `.dvc/config` | **Yes** | Bucket URL, region -- so a third party knows where the data lives |
| `.dvc/config.local` | **No** (gitignored) | Credentials |

The S3 bucket is in `eu-central-1` (Frankfurt), blocks all public access, and is reachable only by a dedicated IAM user (`dvc-user`) with a least-privilege policy scoped to exactly one bucket:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {"Effect": "Allow", "Action": ["s3:ListBucket"],
     "Resource": "arn:aws:s3:::mlops-compliance-classifier-kdubas"},
    {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
     "Resource": "arn:aws:s3:::mlops-compliance-classifier-kdubas/*"}
  ]
}
```

(`ListBucket` acts on the bucket ARN; object operations act on `/*` inside it. Using one ARN for both is a common mistake that produces a confusing 403.)

---

## 5. Two dataset versions

| Tag | md5 of `gold.parquet` | Size | Content |
|---|---|---|---|
| `data-v1` | `0364b94609e6818912a174cbdde55c72` | 105,675 B | First labeling round |
| `data-v2` | `0f5891b999ef5aa1d57046380dabdedf` | 273,467 B | Expanded round |

### Demonstrating it

```bash
git checkout data-v1 && dvc checkout
python -c "import pandas as pd; print(len(pd.read_parquet('data/labeled/gold.parquet')))"
# >> smaller row count

git checkout data-v2 && dvc checkout
python -c "import pandas as pd; print(len(pd.read_parquet('data/labeled/gold.parquet')))"
# >> larger row count

git checkout main && dvc checkout
```

### Reproducing from scratch

```bash
git clone https://github.com/octapricot/compliance-ticket-classifier
cd compliance-ticket-classifier
dvc remote modify --local storage access_key_id     <KEY>
dvc remote modify --local storage secret_access_key <SECRET>
dvc pull
```

---

## 6. Downstream use

This dataset feeds every subsequent homework:

- **HW2** trains on `train.parquet`, evaluates on `test.parquet`
- **HW3** serves the resulting model
- **HW4** uses `train.parquet` as the drift-detection *reference* distribution
- **HW5** runs `dvc pull` inside CI, so a fresh runner reconstructs the exact dataset from its hash before training

---

## 7. Limitations

- **Pre-filter bias.** Tickets with no keyword or exposure signal never reach the annotator, so the model cannot learn to catch them. The dataset systematically under-represents the hardest positives.
- **Single annotator.** No inter-annotator agreement was measured. The schema's consistency is unverified.
- **Label Studio annotations are not themselves DVC-tracked.** The lineage is reproducible from raw to gold, but the intermediate annotation export is not versioned as an artifact.
- **Personal data.** Public Jira/GitHub issues can contain usernames and emails in stack traces. The dataset was not scrubbed. For a production system this would require a minimisation pass and a documented lawful basis.
