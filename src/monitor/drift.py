"""
Data drift detection for the compliance ticket classifier.

Compares a REFERENCE dataset (what the model was trained on) against
CURRENT data (what the live service has actually seen) and reports whether
the input distribution has shifted.

Why this matters here: the model's weights are frozen at training time. If
production tickets drift away from the training distribution, the model
silently gets worse - no error, no alert, just quietly worse predictions.
For a compliance screening tool, that is the dangerous failure mode: the
team believes they are covered when they are not.

Because raw text cannot be compared directly, we compare derived features:
  - text_length   : characters per ticket
  - word_count    : words per ticket
  - prob_relevant : the model's own output distribution (current data only)

Usage:
  python src/monitor/drift.py              # real logged predictions
  python src/monitor/drift.py --simulate   # inject synthetic drift for a demo
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

REFERENCE = Path("data/labeled/train.parquet")
PREDICTIONS = Path("data/monitoring/predictions.jsonl")
OUT_HTML = Path("data/monitoring/drift_report.html")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive comparable numeric features from raw ticket text."""
    df = df.copy()
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text"].str.split().str.len()
    return df


def load_reference() -> pd.DataFrame:
    if not REFERENCE.exists():
        sys.exit(f"Reference data not found at {REFERENCE}. Run: dvc pull")
    df = pd.read_parquet(REFERENCE)
    return add_features(df)[["text_length", "word_count"]]


def load_current() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        sys.exit(
            f"No predictions logged at {PREDICTIONS}.\n"
            "Start the service and send it some requests first."
        )
    rows = [json.loads(line) for line in PREDICTIONS.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit("Prediction log is empty.")
    df = add_features(pd.DataFrame(rows))
    return df[["text_length", "word_count"]]


def simulate_drift(reference: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a deliberately drifted 'current' dataset.

    Real drift takes weeks to appear, which does not fit in a demo. Here we
    synthesise it: we sample the reference and shift the distributions
    (much longer tickets, more words), simulating a team that has moved to
    writing long, detailed tickets. Evidently should detect this.
    """
    drifted = reference.sample(n=min(200, len(reference)), replace=True, random_state=42).copy()
    drifted["text_length"] = (drifted["text_length"] * 2.5 + 300).astype(int)
    drifted["word_count"] = (drifted["word_count"] * 2.2 + 40).astype(int)
    return drifted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulate", action="store_true",
        help="Use synthetically drifted data instead of the real prediction log.",
    )
    args = parser.parse_args()

    reference = load_reference()
    print(f"Reference (training data): {len(reference)} rows")

    if args.simulate:
        current = simulate_drift(reference)
        print(f"Current (SIMULATED drift): {len(current)} rows")
    else:
        current = load_current()
        print(f"Current (logged predictions): {len(current)} rows")

    # Both datasets must share the same schema for a valid comparison.
    definition = DataDefinition(
        numerical_columns=["text_length", "word_count"],
    )
    ref_ds = Dataset.from_pandas(reference, data_definition=definition)
    cur_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(OUT_HTML))
    print(f"\nDrift report written to {OUT_HTML}")

    # Machine-readable summary. Evidently returns two shapes of value:
    #   DriftedColumnsCount -> dict {count, share}
    #   ValueDrift          -> float (the K-S p-value for one column)
    summary = result.dict()
    drift_share = None

    print("\n=== Drift summary ===")
    for metric in summary.get("metrics", []):
        name = metric.get("metric_name", "")
        value = metric.get("value")

        if isinstance(value, dict) and "share" in value:
            drift_share = value["share"]
            print(f"  Drifted columns: {int(value['count'])} "
                  f"({value['share']:.0%} of columns)")
        elif isinstance(value, float):
            # A p-value below the threshold (0.05) means the distributions differ.
            column = name.split("column=")[1].split(",")[0] if "column=" in name else name
            verdict = "DRIFTED" if value < 0.05 else "ok"
            print(f"  {column:<15} p={value:.2e}  -> {verdict}")

    print(f"\nReport: {OUT_HTML}")

    # Exit non-zero when the dataset as a whole is drifted, so this can be
    # used as a gate in CI or wired to an alert.
    if drift_share is not None and drift_share > 0.5:
        print("\n  DATASET DRIFT DETECTED - the model may need retraining.")
        sys.exit(1)

    print("\n No dataset-level drift detected.")


if __name__ == "__main__":
    main()