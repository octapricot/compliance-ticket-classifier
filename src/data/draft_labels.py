"""
AI-assisted labeling: draft a compliance label for each candidate ticket.

For each candidate, Claude acts as a DPO and returns JSON with:
  - label: "relevant" | "not_relevant"
  - confidence: "high" | "medium" | "low"
  - reason: one short sentence

These are DRAFTS. A human (you) reviews/corrects them in Label Studio next.

Input:  data/candidates/candidates.parquet
Output: data/candidates/drafts.parquet   (candidates + AI draft columns)

Progress is saved after every ticket, so a crash mid-run loses nothing:
re-running skips tickets already drafted.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic()

MODEL = "claude-sonnet-4-6"
INPUT = Path("data/candidates/candidates.parquet")
OUTPUT = Path("data/candidates/drafts.parquet")

# How many low-priority candidates to sample (all high-priority are included).
LOW_SAMPLE = 40
MAX_TEXT_CHARS = 2000   # truncate ticket text to control cost & fit context

SYSTEM_PROMPT = """You are an experienced Data Protection Officer (DPO) reviewing \
software-development tickets (from Jira and GitHub) to decide whether each one is \
COMPLIANCE-RELEVANT: i.e. whether a DPO should look at it.

A ticket is RELEVANT if EITHER:
(A) Trigger-by-design: the development task itself creates, stores, processes, \
deletes, or makes automated decisions about personal data (e.g. user data \
export/deletion, analytics/tracking of users, storing user data in logs, \
third-party data integrations, PII handling, automated decision systems).
(B) Incidental exposure: the ticket TEXT itself contains real personal or \
sensitive data (real emails, real user records, credentials) that a DPO would \
want scrubbed.

A ticket is NOT_RELEVANT if it is ordinary engineering with no personal-data \
dimension: logic bugs, performance work, tests, build/config, refactoring.

Important nuances you must apply:
- "profiling" usually means PERFORMANCE profiling here, not GDPR profiling — not relevant by itself.
- "tracking" usually means BUG tracking — not relevant by itself.
- Incidental private/cluster IP addresses (10.x, 172.16-31.x, 192.168.x) or \
generic hostnames in logs are infrastructure noise — NOT relevant on their own.
- A public bug-report email in a boilerplate template is not a data exposure.

Respond with ONLY a JSON object, no other text:
{"label": "relevant" | "not_relevant", "confidence": "high" | "medium" | "low", "reason": "one short sentence"}"""


def draft_one(text: str):
    """Ask Claude to label one ticket. Returns a dict or None on parse failure."""
    msg = client.messages.create(
        model=MODEL,
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Ticket:\n{text[:MAX_TEXT_CHARS]}"}],
    )
    raw = msg.content[0].text.strip()
    try:
        # The model may wrap JSON in ```json fences; strip them.
        cleaned = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  ! could not parse: {raw[:80]}")
        return None


def main():
    cand = pd.read_parquet(INPUT)

    high = cand[cand["priority"] == "high"]
    low = cand[cand["priority"] == "low"].sample(
        min(LOW_SAMPLE, (cand["priority"] == "low").sum()), random_state=42
    )
    work = pd.concat([high, low]).reset_index(drop=True)
    print(f"Labeling {len(work)} candidates ({len(high)} high + {len(low)} low)...")

    # Resume support: if OUTPUT exists, keep already-done rows.
    if OUTPUT.exists():
        done = pd.read_parquet(OUTPUT)
        done_ids = set(done["issue_id"])
        print(f"Resuming: {len(done_ids)} already done.")
    else:
        done = pd.DataFrame()
        done_ids = set()

    results = [done] if len(done) else []

    for i, row in work.iterrows():
        if row["issue_id"] in done_ids:
            continue
        text = row["text"]
        draft = draft_one(text)

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
        # Save after every ticket (crash-safe).
        pd.concat(results, ignore_index=True).to_parquet(OUTPUT, index=False)

        label = rec["ai_label"]
        conf = rec["ai_confidence"]
        print(f"  [{i+1}/{len(work)}] {row['issue_id']}: {label} ({conf})")

        time.sleep(0.3)   # gentle pacing to respect rate limits

    final = pd.read_parquet(OUTPUT)
    print(f"\nDone. {len(final)} drafts saved to {OUTPUT}")
    print(f"\nAI label distribution:\n{final['ai_label'].value_counts()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
