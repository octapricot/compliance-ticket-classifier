"""
Extract v2 labels and merge with existing v1 gold into a combined gold set.

Usage: python src/data/merge_gold_v2.py <v2_export.json>

Reads existing data/labeled/gold.parquet (v1), extracts the v2 export, merges,
dedupes by issue_id (v2 wins on conflict), and overwrites gold.parquet.
The pre-merge v1 gold is preserved as gold_v1_backup.parquet for lineage.
"""
import json
import sys
from pathlib import Path

import pandas as pd

GOLD = Path("data/labeled/gold.parquet")
BACKUP = Path("data/labeled/gold_v1_backup.parquet")


def get_human_label(task):
    anns = task.get("annotations", [])
    if not anns:
        return None
    result = anns[0].get("result", [])
    if not result:
        return None
    try:
        return result[0]["value"]["choices"][0]
    except (KeyError, IndexError):
        return None


def extract(export_path):
    with open(export_path) as f:
        tasks = json.load(f)
    rows = []
    for task in tasks:
        label = get_human_label(task)
        if label is None:
            continue
        d = task.get("data", {})
        rows.append({
            "issue_id": str(d.get("issue_id", "")),
            "source": d.get("source", ""),
            "repo": d.get("repo", ""),
            "priority": d.get("priority", ""),
            "keyword_hits": d.get("keyword_hits", ""),
            "exposure_hits": d.get("exposure_hits", ""),
            "text": d.get("text", ""),
            "label": label,
            "is_synthetic": False,
        })
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/data/merge_gold_v2.py <v2_export.json>")
        sys.exit(1)

    v1 = pd.read_parquet(GOLD)
    v1["issue_id"] = v1["issue_id"].astype(str)
    print(f"v1 gold: {len(v1)} rows")

    # Back up v1 before overwriting (lineage safety).
    if not BACKUP.exists():
        v1.to_parquet(BACKUP, index=False)
        print(f"Backed up v1 gold to {BACKUP}")

    v2 = extract(sys.argv[1])
    print(f"v2 extracted: {len(v2)} rows")

    # Merge, dedupe by issue_id (keep v2 on conflict).
    combined = pd.concat([v1, v2], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="issue_id", keep="last").reset_index(drop=True)
    print(f"Combined: {before} -> {len(combined)} after dedup.")

    combined.to_parquet(GOLD, index=False)

    print(f"\nCombined gold label distribution:\n{combined['label'].value_counts()}")
    print(f"\nPositives by source:\n{combined[combined['label']=='relevant']['source'].value_counts()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
