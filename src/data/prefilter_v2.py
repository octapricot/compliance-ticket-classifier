"""
Pre-filter the v2 raw pull, EXCLUDING already-labeled issues.

Run from the PROJECT ROOT:  python src/data/prefilter_v2.py

Output: data/candidates/candidates_v2.parquet
"""
import sys
from pathlib import Path

# Allow importing prefilter.py (same folder) regardless of current directory.
sys.path.insert(0, str(Path(__file__).parent))
from prefilter import find_keyword_hits, find_fuzzy_hits, find_exposure_hits

import pandas as pd

RAW_V2 = Path("data/raw/issues_sample_v2.parquet")
GOLD = Path("data/labeled/gold.parquet")
OUTPUT = Path("data/candidates/candidates_v2.parquet")


def main():
    df = pd.read_parquet(RAW_V2)
    print(f"Loaded {len(df)} v2 raw issues.")

    gold = pd.read_parquet(GOLD)
    labeled_ids = set(gold["issue_id"].astype(str))
    before = len(df)
    df = df[~df["issue_id"].astype(str).isin(labeled_ids)].copy()
    print(f"Dropped {before - len(df)} already-labeled issues; {len(df)} new remain.")

    df = df.drop_duplicates(subset="issue_id").copy()
    print(f"After dedup within pull: {len(df)} unique new issues.")

    df["text"] = df["title"].fillna("") + "\n\n" + df["body"].fillna("")

    keyword_hits, fuzzy_hits, exposure_hits = [], [], []
    is_candidate, priority = [], []
    for text in df["text"]:
        tl = text.lower()
        kw = find_keyword_hits(tl)
        fz = find_fuzzy_hits(tl)
        ex = find_exposure_hits(text)
        keyword_hits.append(", ".join(kw))
        fuzzy_hits.append(", ".join(fz))
        exposure_hits.append(", ".join(ex))
        has_hostname = "hostname" in ex
        cand = bool(kw or fz or ex)
        is_candidate.append(cand)
        priority.append("" if not cand else ("high" if (kw or fz or has_hostname) else "low"))

    df["keyword_hits"] = keyword_hits
    df["fuzzy_hits"] = fuzzy_hits
    df["exposure_hits"] = exposure_hits
    df["is_candidate"] = is_candidate
    df["priority"] = priority

    candidates = df[df["is_candidate"]].copy()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(OUTPUT, index=False)

    print(f"\nNew candidates: {len(candidates)}")
    print(f"\nBy priority:\n{candidates['priority'].value_counts()}")
    print(f"\nHigh-priority by repo:\n{candidates[candidates['priority']=='high']['repo'].value_counts()}")
    print(f"\nSaved to {OUTPUT}")
    sys.exit(0)


if __name__ == "__main__":
    main()
