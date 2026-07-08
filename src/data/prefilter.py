"""
Pre-filter raw issues to surface COMPLIANCE CANDIDATES for labeling.

Goal: high recall. We'd rather over-catch (reject later during labeling)
than miss a real compliance-relevant ticket.

A ticket becomes a candidate if it trips ANY of:
  1. an exact compliance keyword/phrase,
  2. a fuzzy match against a compliance phrase,
  3. a regex for incidental data exposure (IPs, emails, internal hostnames).

Candidates are given a PRIORITY for labeling:
  - "high": tripped a keyword, fuzzy match, or an internal-hostname exposure.
            These are genuinely compliance-flavored -> label carefully.
  - "low":  tripped ONLY a bare IP or email. In infrastructure projects these
            are mostly network noise (traceroutes, cluster IPs, boilerplate
            emails), but may hide a real leak -> skim these.

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
# PRUNED after a first pass as low-signal "false friends" in engineering
# tickets: authentication/password/credentials (connection plumbing),
# tracking ("bug tracking"), profiling ("performance profiling").
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

# Private/localhost/test IP ranges we treat as noise (RFC1918 + loopback + link-local)
IP_NOISE_PREFIXES = (
    "127.", "0.", "255.", "1.1.1.1",
    "10.",
    "192.168.",
    "169.254.",
    # 172.16.x - 172.31.x are private too
    *[f"172.{i}." for i in range(16, 32)],
)


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
    """Return exposure hits split so callers can tell hostname (high-signal)
    from bare ip/email (noisy)."""
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

    keyword_hits, fuzzy_hits, exposure_hits = [], [], []
    is_candidate, priority = [], []

    for text in df["text"]:
        text_lower = text.lower()
        kw = find_keyword_hits(text_lower)
        fz = find_fuzzy_hits(text_lower)
        ex = find_exposure_hits(text)

        keyword_hits.append(", ".join(kw))
        fuzzy_hits.append(", ".join(fz))
        exposure_hits.append(", ".join(ex))

        has_hostname = "hostname" in ex
        # high-signal if any keyword, fuzzy, or an internal-hostname exposure
        high = bool(kw or fz or has_hostname)
        # candidate at all if anything tripped
        cand = bool(kw or fz or ex)

        is_candidate.append(cand)
        if not cand:
            priority.append("")          # not a candidate
        elif high:
            priority.append("high")
        else:
            priority.append("low")       # exposure-only (bare ip/email)

    df["keyword_hits"] = keyword_hits
    df["fuzzy_hits"] = fuzzy_hits
    df["exposure_hits"] = exposure_hits
    df["is_candidate"] = is_candidate
    df["priority"] = priority

    candidates = df[df["is_candidate"]].copy()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(OUTPUT, index=False)

    n, c = len(df), len(candidates)
    print(f"\nCandidates: {c} / {n}  ({100*c/n:.1f}%)")
    print(f"\nBy priority:\n{candidates['priority'].value_counts()}")
    print(f"\nHigh-priority by source:\n{candidates[candidates['priority']=='high']['source'].value_counts()}")
    print(f"\nHigh-priority by repo:\n{candidates[candidates['priority']=='high']['repo'].value_counts()}")
    print(f"\nSaved candidates to {OUTPUT}")

    sys.exit(0)


if __name__ == "__main__":
    main()
