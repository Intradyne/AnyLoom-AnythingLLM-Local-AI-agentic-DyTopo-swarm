"""Phase 5: LM Studio Prompt Validation - direct to LM Studio API"""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

# Load LM Studio system prompt
prompt_path = os.path.join(os.path.dirname(__file__), "..", "prompts", "lm-studio-system-prompt.md")
with open(prompt_path, "r", encoding="utf-8") as f:
    lms_prompt = f.read()
print(f"Loaded LM Studio prompt: {len(lms_prompt)} chars", flush=True)

def grade_lmstudio(query_id, text, words):
    issues = []
    if query_id == "L2":
        if bh.has_dollar_amount(text):
            return "FAIL", "Fabricated price in LM Studio direct mode"
        if "tavily" not in text.lower() and "tool" not in text.lower():
            issues.append("Did not mention using Tavily or tools for price data")
    else:
        if words > 75:
            if words <= 85:
                issues.append(f"MARGINAL: {words} words (75 limit, 85 tolerance)")
            else:
                issues.append(f"Over word limit: {words} words (max 75)")
        if bh.has_headers(text):
            issues.append("Contains ### headers")
        if bh.has_bullets(text):
            issues.append("Contains bullet lists")

    if query_id == "L3" and "6333" not in text:
        issues.append("Missing port 6333 (may be acceptable without RAG context)")

    if issues:
        grade = "MARGINAL" if all("MARGINAL" in i for i in issues) else "FAIL"
        return grade, "; ".join(issues)
    return "PASS", f"Compliant ({words} words)"

QUERIES = [
    ("L1", "What is DyTopo?"),
    ("L2", "What's the price of gold?"),
    ("L3", "What port does AnythingLLM's Qdrant run on?"),
    ("L4", "What is the trust hierarchy?"),
]

results = []
for qid, query in QUERIES:
    print(f"Sending {qid} to LM Studio...", flush=True)
    raw_text, _ = bh.send_lmstudio(query, lms_prompt)
    text = bh.strip_thinking(raw_text)
    words = len(text.split())
    grade, reason = grade_lmstudio(qid, text, words)
    results.append({
        "id": qid, "query": query,
        "raw_response": raw_text[:1000],
        "stripped_response": text,
        "word_count": words,
        "grade": grade, "reason": reason
    })
    print(f"  {qid}: {words} words, {grade} — {reason}", flush=True)
    time.sleep(3)

out = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarker_phase5_lmstudio.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nPhase 5 done. Results: {out}", flush=True)
