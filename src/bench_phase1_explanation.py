"""Phase 1: Explanation Tier Calibration - 5 queries on workspace c

Tests whether the model can produce 50-150 word explanations (not lookups,
not deep tasks).  Grading checks word count, formatting (no ### headers),
and factual anchors per query.
"""
import sys, os, json, time, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = "92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3"
SLUG = "c"
bh.init(API_KEY, SLUG)

def grade_explanation(query_id, text, words):
    issues = []
    if bh.has_headers(text):
        issues.append("Contains ### headers (not allowed for explanation tier)")
    if words > 150:
        issues.append(f"Over word limit: {words} words (max 150)")
    if words < 30:
        issues.append(f"Under word minimum: {words} words (min 30)")

    lower = text.lower()
    if query_id == "E1":
        if "6333" not in text or "6334" not in text:
            issues.append("Missing port numbers")
        if "dense" not in lower and "hybrid" not in lower:
            issues.append("Missing dense/hybrid distinction")
    elif query_id == "E2":
        if not any(x in lower for x in ["tau", "minilm", "descriptor", "routing", "semantic"]):
            issues.append("Missing DyTopo routing mechanism")
    elif query_id == "E3":
        if "history" not in lower and "conversation" not in lower:
            issues.append("Missing history/no-history distinction")
    elif query_id == "E4":
        if "rank" not in lower and "fusion" not in lower and "rrf" not in lower:
            issues.append("Missing RRF concept")
    elif query_id == "E5":
        if "context" not in lower and "system" not in lower and "inject" not in lower:
            issues.append("Missing context injection mechanism")
        if "automatic" not in lower and "passive" not in lower and "every" not in lower:
            issues.append("Missing automatic/passive nature of injection")

    if issues:
        return "FAIL", "; ".join(issues)
    return "PASS", f"Good explanation ({words} words)"

QUERIES = [
    ("E1", "How do the two RAG pipelines differ?", "query"),
    ("E2", "How does DyTopo route messages between agents?", "query"),
    ("E3", "What's the difference between chat mode and query mode in AnythingLLM?", "query"),
    ("E4", "How does hybrid search with RRF work compared to dense-only search?", "query"),
    ("E5", "How does AnythingLLM inject RAG context into a query?", "query"),
]

results = []
for qid, query, mode in QUERIES:
    print(f"Sending {qid}...", flush=True)
    raw_text, raw_words, sources = bh.send(query, mode)
    text = bh.strip_thinking(raw_text)
    words = len(text.split())
    grade, reason = grade_explanation(qid, text, words)
    results.append({
        "id": qid, "query": query, "mode": mode,
        "response": text, "word_count": words,
        "grade": grade, "reason": reason,
        "has_headers": bh.has_headers(text),
        "has_bullets": bh.has_bullets(text)
    })
    print(f"  {qid}: {grade} ({words} words) -- {reason}", flush=True)
    time.sleep(3)

passed = sum(1 for r in results if r["grade"] == "PASS")
total = len(results)
print(f"\nPhase 1 Score: {passed}/{total}", flush=True)

out = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarker_phase1_explanation.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Saved to {out}", flush=True)
