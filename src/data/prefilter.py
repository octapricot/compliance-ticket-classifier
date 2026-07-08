"""
Pre-filter raw issues to surface COMPLIANCE CANDIDATES for labeling.

Goal: high recall. We'd rather over-catch (reject later during labeling)
than miss a real compliance-relevant ticket.

A ticket becomes a candidate if it trips ANY of:
  1. an exact compliance keyword/phrase,
  2. a fuzzy match against a compliance phrase,
  3. a regex for incidental data exposure (IPs, emails, internal hostnames).

Input:  data/raw/issues_sample.parquet
Output: data/candidates/candidates.parquet
"""
import re
import sys
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

INPUT = Path("data/raw/issues_sample.parquet")
OUTPUT = Path("data/candidates/candidates.parquet")

# ---------------------------------------------------------------------------
# 1. COMPLIANCE KEYWORDS (matched as whole words/phrases, case-insensitive)
# ---------------------------------------------------------------------------
# NOTE: The following keywords were PRUNED after a first pass because they are
# low-signal "false friends" in software-engineering tickets:
#   authentication, password, credentials -> connection/security plumbing,
#       almost never a data-protection concern by themselves.
#   tracking  -> almost always "bug tracking" / "tracking down an issue".
#   profiling -> almost always "performance profiling", not GDPR profiling.
# They inflated the candidate pool with noise without adding real signal.
PRUNED = ["authentication", "password", "credentials", "tracking", "profiling"]

KEYWORDS = [
    # Personal data core
    "personal data", "personal information", "pii", "user data", "customer data",
    "user information", "data subject", "sensitive data",
    # Consent & legal basis
    "consent", "opt-in", "opt-out", "legitimate interest", "legal basis",
    # Data subject rights
    "right to be forgotten", "right to erasure", "delete user", "data deletion",
    "data export", "data portability", "access request", "dsar",
    "anonymize", "anonymise", "pseudonymize", "pseudonymise", "redact",
    # Retention & storage
    "retention", "data retention", "log storage", "purge",
    # Processing & transfer
    "data processing", "third party", "third-party", "processor",
    "sub-processor", "data transfer", "cross-border", "data sharing",
    # Tracking & profiling (specific compliance senses only)
    "behavior monitoring", "cookies", "fingerprint",
    # Automated decisions / AI
    "automated decision", "scoring", "machine learning model", "ai model",
    # Security / privacy adjacent
    "encryption", "gdpr", "hipaa", "ccpa", "privacy", "data breach",
    "audit log", "access control",
]

# ---------------------------------------------------------------------------
# 2. FUZZY PHRASES (multi-word phrases where wording drifts)
# ---------------------------------------------------------------------------
FUZZY_PHRASES = [
    "withdraw consent for personal data",
    "delete user data on request",
    "store personal information in logs",
    "third party data processing agreement",
    "cross border data transfer",
]
FUZZY_THRESHOLD = 82

# ---------------------------------------------------------------------------
# 3. EXPOSURE REGEXES (incidental leakage of real data in the ticket text)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
HOSTNAME_RE = re.compile(r"\b[a-zA-Z0-9.-]+\.(?:internal|corp|local|intra)\.[a-zA-Z0-9.-]+\b")

IP_NOISE_PREFIXES = ("127.", "0.", "10.", "192.168.", "255.", "1.1.1.1")


def find_keyword_hits(text_lower: str):
    hits = []
    for kw in KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
            hits.append(kw)
    return hits


def find_fuzzy_hits(text_lower: str):
    hits = []
    for phrase in FUZZY_PHRASES:
        score = fuzz.partial_ratio(phrase, text_lower)
        if score >= FUZZY_THRESHOLD:
            hits.append(f"{phrase} ({score:.0f})")
    return hits


def find_exposure_hits(text: str):
    hits = []
    if EMAIL_RE.search(text):
        hits.append("email")
    for ip in IP_RE.findall(text):
        if not ip.startswith(IP_NOISE_PREFIXES):
            hits.append(f"ip:{ip}")
            break
    if HOSTNAME_RE.search(text):
        hits.append("hostname")
    return hits


def main():
    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} raw issues.")

    df["text"] = df["title"].fillna("") + "\n\n" + df["body"].fillna("")

    keyword_hits, fuzzy_hits, exposure_hits, is_candidate = [], [], [], []

    for text in df["text"]:
        text_lower = text.lower()
        kw = find_keyword_hits(text_lower)
        fz = find_fuzzy_hits(text_lower)
        ex = find_exposure_hits(text)

        keyword_hits.append(", ".join(kw))
        fuzzy_hits.append(", ".join(fz))
        exposure_hits.append(", ".join(ex))
        is_candidate.append(bool(kw or fz or ex))

    df["keyword_hits"] = keyword_hits
    df["fuzzy_hits"] = fuzzy_hits
    df["exposure_hits"] = exposure_hits
    df["is_candidate"] = is_candidate

    candidates = df[df["is_candidate"]].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(OUTPUT, index=False)

    n, c = len(df), len(candidates)
    print(f"\nCandidates: {c} / {n}  ({100*c/n:.1f}%)")
    print(f"  tripped a keyword:  {(df['keyword_hits'] != '').sum()}")
    print(f"  tripped fuzzy:      {(df['fuzzy_hits'] != '').sum()}")
    print(f"  tripped exposure:   {(df['exposure_hits'] != '').sum()}")
    print(f"\nBy source:\n{candidates['source'].value_counts()}")
    print(f"\nBy repo:\n{candidates['repo'].value_counts()}")
    print(f"\nSaved candidates to {OUTPUT}")

    sys.exit(0)


if __name__ == "__main__":
    main()
