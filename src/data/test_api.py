"""One-off test: confirm the Anthropic API key loads and a call works.
Uses the cheapest model (Haiku). Costs a fraction of a cent.
"""
import os
from dotenv import load_dotenv
from anthropic import Anthropic

# Load variables from .env into the environment.
load_dotenv()

# Confirm the key is present (without printing it!).
key = os.environ.get("ANTHROPIC_API_KEY")
if not key:
    print("ERROR: ANTHROPIC_API_KEY not found. Is .env set up?")
    raise SystemExit(1)
print(f"Key loaded: starts with {key[:10]}..., length {len(key)}")

# The SDK reads ANTHROPIC_API_KEY from the environment automatically.
client = Anthropic()

print("Sending a test message to Claude Haiku...")
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=50,
    messages=[
        {"role": "user", "content": "Reply with exactly: API connection works."}
    ],
)

# The reply text is in response.content[0].text
print("Claude replied:", response.content[0].text)
print("\nSuccess — the API is working.")
