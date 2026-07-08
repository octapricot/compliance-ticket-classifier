"""
Generate synthetic tickets to augment the rare positive class.

Uses your REAL labeled positives as style references (few-shot), asking Claude
to produce NEW tickets in the same terse, technical Jira/GitHub voice. Generates
a mix of:
  - trigger-by-design positives (task processes personal data)
  - incidental-exposure positives (body leaks personal data)
  - hard near-miss NEGATIVES (mentions data concepts but not actually relevant)

All output is tagged is_synthetic=True. These are for TRAINING ONLY — the test
set stays real-only.

Input:  data/labeled/gold.parquet   (for real positive style references)
Output: data/labeled/synthetic.parquet
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
GOLD = Path("data/labeled/gold.parquet")
OUTPUT = Path("data/labeled/synthetic.parquet")

TARGET_POSITIVES = 115   # synthetic positives to generate
TARGET_HARD_NEG = 40     # synthetic hard near-miss negatives
BATCH = 10               # items per API call
N_EXAMPLES = 6           # real positives shown as style refs per call

SYSTEM_PROMPT = """You generate SYNTHETIC software-development tickets (Jira/GitHub \
style) for training a data-protection compliance classifier. Your tickets must read \
like REAL infrastructure/database/dev tickets: terse, technical, jargon-heavy, \
sometimes messy, often WITHOUT obvious privacy keywords. Match the voice of the \
real examples provided. Do NOT write clean, essay-like, or obviously-AI text."""


def build_prompt(examples, kind, n):
    ex_block = "\n\n".join(f"<example>\n{e}\n</example>" for e in examples)

    if kind == "trigger":
        desc = ("COMPLIANCE-RELEVANT because the TASK creates/stores/deletes/processes "
                "personal data or makes automated decisions (e.g. user data export, "
                "deletion, analytics on users, storing user info in logs, third-party "
                "data integrations). Often uses NO explicit privacy words.")
        label = "relevant"
    elif kind == "exposure":
        desc = ("COMPLIANCE-RELEVANT because the ticket BODY incidentally contains real "
                "personal data (a real person's name in a stack trace, a user email in "
                "a repro, a user record) that a DPO would want flagged.")
        label = "relevant"
    else:  # hard_neg
        desc = ("NOT compliance-relevant, but TRICKY: it mentions data/storage/logging/"
                "encryption/user-adjacent concepts, yet is ordinary engineering with no "
                "actual personal-data processing (e.g. performance of a cache, encrypting "
                "replication traffic, a 'user' that means a DB role, profiling = perf).")
        label = "not_relevant"

    return f"""Here are REAL example tickets to match in STYLE (not content):

{ex_block}

Now generate {n} NEW, DIVERSE synthetic tickets that are {desc}

Vary the topic, length, and whether they read as Jira or GitHub. Make them realistic.

Respond with ONLY a JSON array, no other text:
[{{"title": "...", "body": "...", "label": "{label}"}}, ...]"""


def generate_batch(examples, kind, n):
    prompt = build_prompt(examples, kind, n)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        temperature=1.0,          # high temp -> more diverse output
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  ! parse error for {kind}: {cleaned[:80]}")
        return []


def main():
    gold = pd.read_parquet(GOLD)
    real_pos = gold[gold["label"] == "relevant"]["text"].tolist()
    print(f"Using {len(real_pos)} real positives as style references.")

    plan = (
        [("trigger", TARGET_POSITIVES // 2)]
        + [("exposure", TARGET_POSITIVES - TARGET_POSITIVES // 2)]
        + [("hard_neg", TARGET_HARD_NEG)]
    )

    rows = []
    for kind, total in plan:
        made = 0
        print(f"\nGenerating {total} '{kind}' tickets...")
        while made < total:
            n = min(BATCH, total - made)
            # rotate through real examples for variety
            examples = pd.Series(real_pos).sample(
                min(N_EXAMPLES, len(real_pos))
            ).tolist()
            batch = generate_batch(examples, kind, n)
            for item in batch:
                if "title" in item and "body" in item and "label" in item:
                    text = f"{item['title']}\n\n{item['body']}"
                    rows.append({
                        "issue_id": f"synth_{kind}_{len(rows)}",
                        "source": "synthetic",
                        "repo": "synthetic",
                        "priority": "",
                        "keyword_hits": "",
                        "exposure_hits": "",
                        "text": text,
                        "label": item["label"],
                        "is_synthetic": True,
                        "synth_kind": kind,
                    })
                    made += 1
            print(f"  {made}/{total}")
            time.sleep(0.5)
            # save progress each batch
            pd.DataFrame(rows).to_parquet(OUTPUT, index=False)

    df = pd.read_parquet(OUTPUT)
    print(f"\nDone. {len(df)} synthetic items saved to {OUTPUT}")
    print(f"\nBy label:\n{df['label'].value_counts()}")
    print(f"\nBy kind:\n{df['synth_kind'].value_counts()}")
    sys.exit(0)


if __name__ == "__main__":
    main()
