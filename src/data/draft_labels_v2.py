"""
AI-draft labels for the v2 candidate round.

Reuses draft_one() from draft_labels.py. Targets ~200 new candidates:
all high-priority + a sample of low-priority.

Input:  data/candidates/candidates_v2.parquet
Output: data/candidates/drafts_v2.parquet  (crash-safe, resumable)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from draft_labels import draft_one   # reuse the same DPO prompt + parser

import pandas as pd

INPUT = Path("data/candidates/candidates_v2.parquet")
OUTPUT = Path("data/candidates/drafts_v2.parquet")

TARGET_TOTAL = 200
SEED = 42


def main():
    cand = pd.read_parquet(INPUT)
    high = cand[cand["priority"] == "high"]
    low = cand[cand["priority"] == "low"]

    n_low = min(len(low), TARGET_TOTAL - len(high))
    low_sample = low.sample(n_low, random_state=SEED)
    work = pd.concat([high, low_sample]).reset_index(drop=True)
    print(f"Drafting {len(work)} candidates ({len(high)} high + {n_low} low)...")

    if OUTPUT.exists():
        done = pd.read_parquet(OUTPUT)
        done_ids = set(done["issue_id"].astype(str))
        results = [done]
        print(f"Resuming: {len(done_ids)} already done.")
    else:
        done_ids = set()
        results = []

    for i, row in work.iterrows():
        if str(row["issue_id"]) in done_ids:
            continue
        draft = draft_one(row["text"])
        rec = row.to_dict()
        if draft:
            rec["ai_label"] = draft.get("label", "")
            rec["ai_confidence"] = draft.get("confidence", "")
            rec["ai_reason"] = draft.get("reason", "")
        else:
            rec["ai_label"] = "PARSE_ERROR"
            rec["ai_confidence"] = ""
            rec["ai_reason"] = ""
        results.append(pd.DataFrame([rec]))
        pd.concat(results, ignore_index=True).to_parquet(OUTPUT, index=False)
        print(f"  [{i+1}/{len(work)}] {row['issue_id']}: {rec['ai_label']} ({rec['ai_confidence']})")
        time.sleep(0.3)

    final = pd.read_parquet(OUTPUT)
    print(f"\nDone. {len(final)} drafts saved.")
    print(f"\nAI label distribution:\n{final['ai_label'].value_counts()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
