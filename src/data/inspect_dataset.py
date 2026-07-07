"""Quick inspection of the hankzhwang/issues dataset structure.
Throwaway helper: shows us the real column names before we write the puller.
"""
from datasets import load_dataset

# Stream one split so we don't download the whole thing.
# streaming=True means: read rows on demand, don't pull all 1.4 GB.
ds = load_dataset(
    "hankzhwang/issues",
    "issues",                       # the subset (not 'comments')
    split="jira__mongodb__SERVER",  # one Jira split
    streaming=True,
)

# Grab the very first row and look at it.
first = next(iter(ds))

print("COLUMN NAMES:")
for key in first.keys():
    print(f"  - {key}")

print("\nFIRST ROW (truncated preview):")
for key, value in first.items():
    preview = str(value)[:200]   # first 200 chars only, so it's readable
    print(f"\n[{key}]\n{preview}")
