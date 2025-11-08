from anthropic import Anthropic
from dotenv import load_dotenv
import os
from pathlib import Path

# ✅ Explicitly tell Python where .env is located
dotenv_path = Path(__file__).parent / ".env"
print(f"🔍 Loading .env from: {dotenv_path.resolve()}")

# Load the environment variables
load_dotenv(dotenv_path=dotenv_path)

# Verify the key
key = os.getenv("ANTHROPIC_API_KEY")
if not key:
    raise EnvironmentError("❌ Anthropic API key not found in .env file.")
else:
    print("✅ Key loaded:", key[:10] + "...")

# Initialize the client
client = Anthropic(api_key=key)

print("🚀 Sending test message to Claude...")

# Make a simple API call
resp = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=50,
    messages=[{"role": "user", "content": "Hello Claude! Please reply with OK."}]
)

print("✅ Claude replied:", resp.content[0].text)
