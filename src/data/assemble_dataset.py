"""
Assemble the final training-ready dataset.

Combines:
  - REAL positives (from gold.parquet, human-labeled)
  - REAL negatives: labeled 'not_relevant' from gold + fresh random non-candidates
  - SYNTHETIC positives + synthetic hard-negatives (training only)

CRITICAL: the held-out TEST set is REAL-ONLY. Synthetics are added to TRAIN only.
Positives use a LARGER test fraction than negatives, because real positives are
scarce and we want enough of them in the test set for a less-noisy evaluation.

Outputs:
  data/labeled/train.parquet
  data/labeled/test.parquet
"""
import sys
from pathlib import Path

import pandas as pd

GOLD = Path("data/labeled/gold.parquet")
SYNTH = Path("data/labeled/synthetic.parquet")
RAW = Path("data/raw/issues_sample.parquet")
CANDIDATES = Path("data/candidates/candidates.parquet")

TRAIN_OUT = Path("data/labeled/train.parquet")
TEST_OUT = Path("data/labeled/test.parquet")

N_EASY_NEG = 150
# Different test fractions per class: hold out more positives for a better test.
TEST_FRAC = {"relevant": 0.40, "not_relevant": 0.25}
SEED = 42


def main():
    gold = pd.read_parquet(GOLD)
    synth = pd.read_parquet(SYNTH)
    raw = pd.read_parquet(RAW)
    cand = pd.read_parquet(CANDIDATES)

    raw["text"] = raw["title"].fillna("") + "\n\n" + raw["body"].fillna("")
    candidate_ids = set(cand["issue_id"])
    non_candidates = raw[~raw["issue_id"].isin(candidate_ids)].copy()
    easy_neg = non_candidates.sample(N_EASY_NEG, random_state=SEED)
    easy_neg = pd.DataFrame({
        "issue_id": easy_neg["issue_id"],
        "source": easy_neg["source"],
        "repo": easy_neg["repo"],
        "text": easy_neg["text"],
        "label": "not_relevant",
        "is_synthetic": False,
    })
    print(f"Real: {len(gold)} gold-labeled + {len(easy_neg)} easy negatives added.")

    real = pd.concat([
        gold[["issue_id", "source", "repo", "text", "label", "is_synthetic"]],
        easy_neg,
    ], ignore_index=True)

    # Stratified split with a per-class test fraction.
    test_parts, train_parts = [], []
    for lbl, grp in real.groupby("label"):
        grp = grp.sample(frac=1, random_state=SEED)
        n_test = int(len(grp) * TEST_FRAC[lbl])
        test_parts.append(grp.iloc[:n_test])
        train_parts.append(grp.iloc[n_test:])

    test = pd.concat(test_parts, ignore_index=True)
    real_train = pd.concat(train_parts, ignore_index=True)

    synth_clean = synth[["issue_id", "source", "repo", "text", "label", "is_synthetic"]]
    train = pd.concat([real_train, synth_clean], ignore_index=True)

    train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test = test.sample(frac=1, random_state=SEED).reset_index(drop=True)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_OUT, index=False)
    test.to_parquet(TEST_OUT, index=False)

    print(f"\n=== TRAIN ({len(train)}) ===")
    print(train["label"].value_counts())
    print(f"synthetic in train: {train['is_synthetic'].sum()}")
    print(f"\n=== TEST ({len(test)}) — REAL ONLY ===")
    print(test["label"].value_counts())
    print(f"synthetic in test: {test['is_synthetic'].sum()}  (must be 0)")
    sys.exit(0)


if __name__ == "__main__":
    main()
