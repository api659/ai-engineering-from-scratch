import json
import os
import urllib.request

PROMPT = "What is a neural network in one sentence?"


def load_dotenv():
    try:
        with open(".env", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    except FileNotFoundError:
        pass


def call_gemini():
    model = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first")
        return

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": PROMPT}]}],
        "generationConfig": {"maxOutputTokens": 256},
    }).encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"Gemini request failed ({error.code}): {details[:300]}")
        return

    text = result["candidates"][0]["content"]["parts"][0]["text"]
    usage = result.get("usageMetadata", {})
    print(f"Gemini response: {text}")
    print(
        "Tokens used: "
        f"{usage.get('promptTokenCount', '?')} in, "
        f"{usage.get('candidatesTokenCount', '?')} out"
    )


if __name__ == "__main__":
    load_dotenv()
    print("=== Gemini API Call ===\n")
    call_gemini()
