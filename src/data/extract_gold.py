"""
Extract final human labels from the Label Studio export into gold data.

Reads the exported JSON, pulls each ticket's confirmed human label from
`annotations`, reunites it with the ticket text + metadata, and saves the
gold labeled dataset.

Usage: python src/data/extract_gold.py <path_to_export.json>

Output: data/labeled/gold.parquet
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUTPUT = Path("data/labeled/gold.parquet")


def get_human_label(task):
    """Return the confirmed human label, or None if unlabeled."""
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/data/extract_gold.py <export.json>")
        sys.exit(1)

    export_path = sys.argv[1]
    with open(export_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks from export.")

    rows = []
    unlabeled = 0
    for task in tasks:
        label = get_human_label(task)
        if label is None:
            unlabeled += 1
            continue
        d = task.get("data", {})
        rows.append({
            "issue_id": d.get("issue_id", ""),
            "source": d.get("source", ""),
            "repo": d.get("repo", ""),
            "priority": d.get("priority", ""),
            "keyword_hits": d.get("keyword_hits", ""),
            "exposure_hits": d.get("exposure_hits", ""),
            "text": d.get("text", ""),
            "label": label,
            "is_synthetic": False,   # these are all REAL, human-labeled
        })

    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT, index=False)

    print(f"\nSaved {len(df)} gold-labeled tickets ({unlabeled} were unlabeled/skipped).")
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
    print(f"\nPositives by source:\n{df[df['label']=='relevant']['source'].value_counts()}")
    print(f"\nPositives by repo:\n{df[df['label']=='relevant']['repo'].value_counts()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
