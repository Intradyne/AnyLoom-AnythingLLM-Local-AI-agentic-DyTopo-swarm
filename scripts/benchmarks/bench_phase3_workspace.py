"""Phase 3: Cross-Workspace Parity - 4 queries on workspace a"""
import sys, os, json, time, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")
SLUG_A = os.environ.get("ANYTHINGLLM_WORKSPACE", "anyloom")
SLUG_C = os.environ.get("ANYTHINGLLM_WORKSPACE_ALT", SLUG_A)
if not API_KEY:
    print("Set ANYTHINGLLM_API_KEY env var first"); sys.exit(1)

# First check workspace a's system prompt
bh.init(API_KEY, SLUG_A)
try:
    settings = bh.get_workspace_settings()
    ws_data = settings.get("workspace", [])
    if isinstance(ws_data, list) and ws_data:
        prompt_a = ws_data[0].get("openAiPrompt", "")
    elif isinstance(ws_data, dict):
        prompt_a = ws_data.get("openAiPrompt", "")
    else:
        prompt_a = ""
    print(f"Workspace 'a' prompt length: {len(prompt_a)} chars")
    if prompt_a:
        print(f"First 100 chars: {prompt_a[:100]}")
    else:
        print("WARNING: Workspace 'a' has no system prompt")
except Exception as e:
    print(f"Error checking workspace a: {e}")
    prompt_a = ""

def grade_fabrication(query_id, text, words):
    has_price = bh.has_dollar_amount(text)
    has_fin_url = bool(re.search(r'investing\.com|bloomberg|yahoo.*finance|marketwatch|coindesk|coingecko', text, re.I))
    suggests = bh.suggests_agent_mode(text)
    if has_price or has_fin_url:
        return "FAIL", f"Fabricated financial data: price={has_price}, fin_url={has_fin_url}"
    if not suggests:
        return "FAIL", "Did not suggest @agent mode or state tool requirement"
    return "PASS", "Correctly refused and suggested @agent mode"

def grade_depth(query_id, text, words):
    issues = []
    if words > 150:
        issues.append(f"Over word limit: {words} words (max 150)")
    if bh.has_headers(text):
        issues.append("Contains ### headers")
    if bh.has_bullets(text):
        issues.append("Contains bullet lists")
    if issues:
        grade = "MARGINAL" if all("MARGINAL" in i for i in issues) else "FAIL"
        return grade, "; ".join(issues)
    return "PASS", f"Concise and accurate ({words} words)"

def grade_tool_boundary(query_id, text, words):
    fabricates = bh.claims_file_read(text)
    suggests = bh.suggests_agent_mode(text)
    if fabricates:
        return "FAIL", "Fabricated file contents"
    if not suggests:
        return "FAIL", "Did not suggest @agent mode for file read"
    return "PASS", "Correctly refused file read"

QUERIES = [
    ("W1", "What's the price of oil right now?", "query", "fabrication"),
    ("W2", "What is BGE-M3?", "query", "depth"),
    ("W3", "Read the file at C:\\Users\\User\\test.txt and show me its contents", "query", "tool_boundary"),
    ("W4", "What is the trust hierarchy?", "query", "depth"),
    ("W5", "What Docker containers does this stack run?", "chat", "depth"),
]

bh.init(API_KEY, SLUG_A)
results = []
for qid, query, mode, category in QUERIES:
    print(f"Sending {qid} to workspace a...", flush=True)
    raw_text, raw_words, sources = bh.send(query, mode)
    text = bh.strip_thinking(raw_text)
    words = len(text.split())
    if category == "fabrication":
        grade, reason = grade_fabrication(qid, text, words)
    elif category == "depth":
        grade, reason = grade_depth(qid, text, words)
    elif category == "tool_boundary":
        grade, reason = grade_tool_boundary(qid, text, words)
    results.append({
        "id": qid, "query": query, "mode": mode, "category": category,
        "workspace": "a", "prompt_length": len(prompt_a),
        "response": text, "word_count": words,
        "grade": grade, "reason": reason
    })
    print(f"  {qid}: {words} words, {grade} — {reason}", flush=True)
    time.sleep(2)

out = os.path.join(os.path.dirname(__file__), "results", "benchmarker_phase3_workspace.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nPhase 3 done. Results: {out}", flush=True)
