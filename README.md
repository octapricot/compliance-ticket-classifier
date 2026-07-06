# Compliance Ticket Classifier

A machine-learning system that flags software-development tickets (Jira / GitHub
issues) as **compliance-relevant** (e.g. GDPR, EU AI Act) or not.

This repository serves two courses:
- **Deep Learning capstone** — the model itself (baseline + fine-tuned transformer)
  and a full development cycle: data → training → evaluation.
- **MLOps course** — the full lifecycle around that model: data labeling and
  versioning, experiment tracking, a model registry, inference serving,
  monitoring, and CI/CD.

## Problem

Development teams routinely create tickets that trigger data-protection obligations
without using privacy keywords (e.g. "add log storage microservice" may create a
system subject to data-protection-by-design requirements). This project trains a
classifier to surface such tickets automatically.

## Data

Raw text: public Jira/GitHub issues from the Hugging Face dataset
`hankzhwang/issues`. The dataset has no compliance labels; labeling those
(informed by DPO domain expertise) is part of this project.

## Project structure

- `data/` - raw, candidate, and labeled datasets (DVC-tracked, not in Git)
- `src/data/` - download, pre-filter, and export-for-labeling scripts
- `src/train/` - baseline and transformer training
- `src/serve/` - inference API (FastAPI)
- `src/monitor/` - metrics and monitoring
- `models/` - trained models (registry-tracked, not in Git)
- `notebooks/` - exploration
- `config/` - paths, thresholds, hyperparameters
- `.github/workflows/` - CI/CD pipelines
- `docs/` - per-homework write-ups and the capstone report

## Setup

`python3 -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`

## Status

Project scaffolding complete. Data labeling in progress.
