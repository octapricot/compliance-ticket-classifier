"""Convert v2 drafts into Label Studio import format."""
import json
import sys
from pathlib import Path
import pandas as pd

INPUT = Path("data/candidates/drafts_v2.parquet")
OUTPUT = Path("data/candidates/label_studio_tasks_v2.json")
LABEL_MAP = {"relevant": "relevant", "not_relevant": "not_relevant"}


def main():
    df = pd.read_parquet(INPUT)
    tasks = []
    for _, row in df.iterrows():
        data = {
            "text": row["text"][:3000],
            "source": row.get("source", ""),
            "repo": row.get("repo", ""),
            "priority": row.get("priority", ""),
            "keyword_hits": row.get("keyword_hits", ""),
            "exposure_hits": row.get("exposure_hits", ""),
            "ai_reason": row.get("ai_reason", ""),
            "ai_confidence": row.get("ai_confidence", ""),
            "issue_id": str(row.get("issue_id", "")),
        }
        task = {"data": data}
        ai_label = row.get("ai_label", "")
        if ai_label in LABEL_MAP:
            task["predictions"] = [{
                "result": [{
                    "from_name": "label", "to_name": "text",
                    "type": "choices",
                    "value": {"choices": [LABEL_MAP[ai_label]]},
                }],
            }]
        tasks.append(task)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(tasks, f, indent=2)
    n_pred = sum(1 for t in tasks if "predictions" in t)
    print(f"Wrote {len(tasks)} tasks ({n_pred} with AI pre-annotations) to {OUTPUT}")
    sys.exit(0)


if __name__ == "__main__":
    main()
