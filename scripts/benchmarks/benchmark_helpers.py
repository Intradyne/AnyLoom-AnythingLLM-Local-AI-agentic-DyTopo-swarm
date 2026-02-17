import requests, json, re, time, sys, os

# Suppress requests urllib3 warnings
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

BASE = "http://localhost:3001/api/v1"
KEY  = ""
SLUG = ""
HDRS = {}

def init(api_key, workspace_slug):
    global KEY, SLUG, HDRS
    KEY  = api_key
    SLUG = workspace_slug
    HDRS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

def send(message, mode="query"):
    """Send a message to the AnythingLLM workspace. Returns (text, word_count, sources)."""
    r = requests.post(
        f"{BASE}/workspace/{SLUG}/chat",
        headers=HDRS,
        json={"message": message, "mode": mode},
        timeout=120
    )
    r.raise_for_status()
    data = r.json()
    text = data.get("textResponse", "")
    words = len(text.split())
    sources = data.get("sources", [])
    return text, words, sources

def has_dollar_amount(text):
    return bool(re.search(r'\$[\d,]+\.?\d*', text))

def has_headers(text):
    return bool(re.search(r'^#{1,4}\s', text, re.MULTILINE))

def has_bullets(text):
    return bool(re.search(r'^\s*[-*]\s', text, re.MULTILINE))

def has_url(text):
    return bool(re.search(r'https?://', text))

def suggests_agent_mode(text):
    lower = text.lower()
    return ("@agent" in lower or "agent mode" in lower or
            "tool call" in lower or "requires a live" in lower or
            ("cannot" in lower and "tool" in lower))

def claims_file_read(text):
    indicators = ["```python", "```json", "def ", "import ", "class ",
                   "here are the first", "the file contains", "the contents"]
    lower = text.lower()
    return any(ind.lower() in lower for ind in indicators)

def update_topn(new_topn):
    r = requests.post(
        f"{BASE}/workspace/{SLUG}/update",
        headers=HDRS,
        json={"topN": new_topn}
    )
    r.raise_for_status()
    return r.json()

def get_workspace_settings():
    r = requests.get(f"{BASE}/workspace/{SLUG}", headers=HDRS)
    r.raise_for_status()
    return r.json()

def send_llm(message, system_prompt):
    """Send a message directly to llama.cpp API with a custom system prompt."""
    r = requests.post(
        "http://localhost:8008/v1/chat/completions",
        json={
            "model": "qwen3-30b-a3b-instruct-2507",
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": message}
            ]
        },
        timeout=300
    )
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    return text, len(text.split())

def strip_thinking(text):
    """Remove Qwen3 thinking tags before grading."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
