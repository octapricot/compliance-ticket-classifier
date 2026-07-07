"""
Pull a working sample of issues from the hankzhwang/issues dataset.

Sources (to match the capstone's dual Jira + GitHub design):
  - jira__mongodb__SERVER          (Jira)
  - github__cockroachdb__cockroach (GitHub)

Output: data/raw/issues_sample.parquet
"""
import sys
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# ---- Settings ----
SPLITS = {
    "jira__mongodb__SERVER": "jira",
    "github__cockroachdb__cockroach": "github",
}
PER_SPLIT = 2500
OUTPUT = Path("data/raw/issues_sample.parquet")

KEEP = ["issue_id", "number", "title", "body", "state",
        "created_at", "comments_count"]


def pull_split(split_name: str, source_label: str, limit: int):
    """Stream `limit` rows from one split and return them as a list of dicts."""
    print(f"Pulling {limit} issues from '{split_name}' ...")
    ds = load_dataset(
        "hankzhwang/issues",
        "issues",
        split=split_name,
        streaming=True,
    )

    rows = []
    for i, record in enumerate(ds):
        if i >= limit:
            break
        row = {col: record.get(col) for col in KEEP}
        row["source"] = source_label
        rows.append(row)
    print(f"  got {len(rows)} rows")
    return rows


def main():
    all_rows = []
    for split_name, source_label in SPLITS.items():
        all_rows.extend(pull_split(split_name, source_label, PER_SPLIT))

    df = pd.DataFrame(all_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\nSaved {len(df)} issues to {OUTPUT}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    # Force a clean shutdown so the open streaming connection doesn't linger.
    sys.exit(0)


if __name__ == "__main__":
    main()
