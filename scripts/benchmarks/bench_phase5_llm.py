"""Phase 5: LLM Prompt Validation - direct to llama.cpp API"""
import sys, os, json, time, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

# Load LLM system prompt
prompt_path = os.path.join(os.path.dirname(__file__), "..", "..", "prompts", "llm-system-prompt.md")
with open(prompt_path, "r", encoding="utf-8") as f:
    llm_prompt = f.read()
print(f"Loaded LLM prompt: {len(llm_prompt)} chars", flush=True)

def grade_llm_direct(query_id, text, words):
    issues = []
    if query_id == "L2":
        if bh.has_dollar_amount(text):
            return "FAIL", "Fabricated price in LLM direct mode"
        if "tavily" not in text.lower() and "tool" not in text.lower():
            issues.append("Did not mention using Tavily or tools for price data")
    else:
        if words > 150:
            issues.append(f"Over word limit: {words} words (max 150)")
        if bh.has_headers(text):
            issues.append("Contains ### headers")
        if bh.has_bullets(text):
            issues.append("Contains bullet lists")

    if query_id == "L3" and "6333" not in text:
        issues.append("Missing port 6333 (may be acceptable without RAG context)")

    if query_id == "L5":
        # Tool boundary: should refuse web search without MCP tools
        if bh.suggests_agent_mode(text) or "tavily" in text.lower() or "tool" in text.lower():
            return "PASS", f"Correctly identified tool requirement ({words} words)"
        return "FAIL", "Did not identify that web search requires tools"

    if issues:
        grade = "MARGINAL" if all("MARGINAL" in i for i in issues) else "FAIL"
        return grade, "; ".join(issues)
    return "PASS", f"Compliant ({words} words)"

QUERIES = [
    ("L1", "What is DyTopo?"),
    ("L2", "What's the price of gold?"),
    ("L3", "What port does AnythingLLM's Qdrant run on?"),
    ("L4", "What is the trust hierarchy?"),
    ("L5", "Search the web for the latest Qdrant release notes"),
]

results = []
for qid, query in QUERIES:
    print(f"Sending {qid} to LLM...", flush=True)
    raw_text, _ = bh.send_llm(query, llm_prompt)
    text = bh.strip_thinking(raw_text)
    words = len(text.split())
    grade, reason = grade_llm_direct(qid, text, words)
    results.append({
        "id": qid, "query": query,
        "raw_response": raw_text[:1000],
        "stripped_response": text,
        "word_count": words,
        "grade": grade, "reason": reason
    })
    print(f"  {qid}: {words} words, {grade} — {reason}", flush=True)
    time.sleep(3)

out = os.path.join(os.path.dirname(__file__), "results", "benchmarker_phase5_llm.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nPhase 5 done. Results: {out}", flush=True)
