"""
Convert AI-drafted candidates into Label Studio import format (JSON).

Each ticket becomes a "task" with:
  - data: the text + context shown to the reviewer
  - predictions: the AI's draft label, pre-filled for review

Input:  data/candidates/drafts.parquet
Output: data/candidates/label_studio_tasks.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

INPUT = Path("data/candidates/drafts.parquet")
OUTPUT = Path("data/candidates/label_studio_tasks.json")

# Map our AI labels to the choice values shown in Label Studio.
LABEL_MAP = {
    "relevant": "relevant",
    "not_relevant": "not_relevant",
    # PARSE_ERROR or blanks -> no pre-annotation
}


def main():
    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} drafted candidates.")

    tasks = []
    for _, row in df.iterrows():
        # Data shown to you during review.
        data = {
            "text": row["text"][:3000],          # cap display length
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

        # If the AI produced a usable label, attach it as a pre-annotation.
        ai_label = row.get("ai_label", "")
        if ai_label in LABEL_MAP:
            task["predictions"] = [{
                "result": [{
                    "from_name": "label",
                    "to_name": "text",
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
