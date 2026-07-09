"""
Pull a LARGER sample for the v2 labeling round.

Pulls more issues per split than v1 (to reach deeper into each repo and surface
NEW candidates). Deduplication against already-labeled issues happens later,
in the v2 prefilter step.

Output: data/raw/issues_sample_v2.parquet
"""
import sys
from datasets import load_dataset
import pandas as pd
from pathlib import Path

# Larger counts than v1 (v1 used 2500 each). Pulling from the top again is fine
# because we DEDUPE by issue_id downstream — the extra rows beyond 2500 are new.
SPLITS = {
    "jira__mongodb__SERVER":          ("jira", 7000),
    "github__cockroachdb__cockroach": ("github", 7000),
    "github__etcd_io__etcd":          ("github", 7000),
    "github__microsoft__WSL":         ("github", 7000),
}
OUTPUT = Path("data/raw/issues_sample_v2.parquet")

KEEP = ["issue_id", "number", "title", "body", "state",
        "created_at", "comments_count"]


def pull_split(split_name, source_label, limit):
    print(f"Pulling {limit} issues from '{split_name}' ...")
    ds = load_dataset("hankzhwang/issues", "issues",
                      split=split_name, streaming=True)
    rows = []
    for i, record in enumerate(ds):
        if i >= limit:
            break
        row = {col: record.get(col) for col in KEEP}
        row["source"] = source_label
        row["repo"] = split_name
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
    sys.exit(0)


if __name__ == "__main__":
    main()
