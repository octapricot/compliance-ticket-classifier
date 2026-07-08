"""
Pull a working sample of issues from the hankzhwang/issues dataset.

Sources (Jira + multiple GitHub repos for dual-source, varied-style coverage):
  - jira__mongodb__SERVER          (Jira)      2500
  - github__cockroachdb__cockroach (GitHub)    2500
  - github__etcd_io__etcd          (GitHub)    2500
  - github__microsoft__WSL         (GitHub)    2500

GitHub is over-weighted in volume because its tickets are short and often
empty, so more are needed to surface a comparable candidate count.

Output: data/raw/issues_sample.parquet
"""
import sys
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# ---- Settings ----
# Each split: (source_label, how_many_to_pull)
SPLITS = {
    "jira__mongodb__SERVER":          ("jira", 2500),
    "github__cockroachdb__cockroach": ("github", 2500),
    "github__etcd_io__etcd":          ("github", 2500),
    "github__microsoft__WSL":         ("github", 2500),
}
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
        row["repo"] = split_name          # remember the exact repo too
        rows.append(row)
    print(f"  got {len(rows)} rows")
    return rows


def main():
    all_rows = []
    for split_name, (source_label, limit) in SPLITS.items():
        all_rows.extend(pull_split(split_name, source_label, limit))

    df = pd.DataFrame(all_rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\nSaved {len(df)} issues to {OUTPUT}")
    print(f"By source: {df['source'].value_counts().to_dict()}")
    print(f"By repo:   {df['repo'].value_counts().to_dict()}")

    sys.exit(0)


if __name__ == "__main__":
    main()
