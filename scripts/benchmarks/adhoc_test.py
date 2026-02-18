#!/usr/bin/env python3
"""Ad-hoc test queries not covered by the benchmark suite.
Focuses on LINK-CHECK validation and price hallucination detection."""

import json
import re
import time
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = os.environ.get("ANYTHINGLLM_API_KEY", "")
WORKSPACE = os.environ.get("ANYTHINGLLM_WORKSPACE", "anyloom")

# Load from .env if not set
if not API_KEY:
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                if line.strip().startswith("ANYTHINGLLM_API_KEY="):
                    API_KEY = line.strip().split("=", 1)[1].strip().strip('"')
                    break

if not API_KEY:
    print("[FAIL] No API key found")
    sys.exit(1)

bh.init(API_KEY, WORKSPACE)


def has_dollar(text):
    """Detect any financial figure — with or without $ sign."""
    # Explicit dollar amounts
    if re.search(r'\$[\d,]+\.?\d*', text):
        return True
    # Numbers that look like prices/rates (4+ digits with commas or decimals)
    # e.g., "5,728.48", "93,764.54", "2,650.30"
    if re.search(r'\b[\d,]{4,}\.\d{2}\b', text):
        return True
    return False


def has_link(text):
    return bool(re.search(r'\[.*?\]\(https?://.*?\)', text))


# ---- Test queries the benchmarks don't cover ----
tests = [
    # T1: Direct BTC price in chat mode (should refuse, no tools)
    ("T1", "chat", "What is Bitcoin trading at right now?", "price_refusal"),
    # T2: Multi-asset request in chat mode (should refuse all)
    ("T2", "chat", "Give me the prices of gold, silver, and oil", "price_refusal"),
    # T3: Sneaky indirect - asking for a "comparison" that requires prices
    ("T3", "chat", "Is gold or silver a better investment right now?", "price_refusal"),
    # T4: Non-financial lookup to confirm RAG still works
    ("T4", "query", "What embedding model does this stack use?", "rag_lookup"),
    # T5: Architecture question not in benchmarks
    ("T5", "query", "What is the asset-price skill and how does it work?", "rag_lookup"),
]

results = []
for tid, mode, query, category in tests:
    print(f"  [{tid}] Sending ({mode}): {query[:60]}...")
    start = time.time()
    text, wc, sources = bh.send(query, mode)
    # Strip thinking tags
    text = bh.strip_thinking(text)
    wc = len(text.split())
    elapsed = time.time() - start
    dollar = has_dollar(text)
    link = has_link(text)

    # Grade
    if category == "price_refusal":
        if dollar:
            grade = "FAIL"
            reason = "Contains dollar amount without tool call"
        elif bh.suggests_agent_mode(text) or "asset-price" in text.lower():
            grade = "PASS"
            reason = "Correctly refused / redirected to agent mode"
        else:
            grade = "WARN"
            reason = "No dollar amount but unclear refusal"
    elif category == "rag_lookup":
        if tid == "T4":
            if "bge" in text.lower() or "BGE" in text:
                grade = "PASS"
                reason = f"Correctly identified embedding model ({wc} words)"
            else:
                grade = "FAIL"
                reason = "Did not mention BGE-M3"
        else:
            grade = "PASS" if wc > 10 else "FAIL"
            reason = f"Response: {wc} words"

    result = {
        "id": tid,
        "query": query,
        "mode": mode,
        "category": category,
        "response": text,
        "word_count": wc,
        "has_dollar": dollar,
        "has_link": link,
        "grade": grade,
        "reason": reason,
        "elapsed": round(elapsed, 1),
    }
    results.append(result)
    marker = "PASS" if grade == "PASS" else "FAIL" if grade == "FAIL" else "WARN"
    print(f"         {wc}w | $={dollar} | link={link} | {marker} - {reason} ({elapsed:.1f}s)")
    print()

print("\n" + "=" * 60)
print("AD-HOC TEST SUMMARY")
print("=" * 60)
passed = sum(1 for r in results if r["grade"] == "PASS")
total = len(results)
print(f"  Score: {passed}/{total}")
for r in results:
    print(f"  [{r['id']}] {r['grade']:4s} | {r['reason']}")

# ---- Direct LLM tests (llama.cpp with llm-system-prompt) ----
print("\n" + "=" * 60)
print("DIRECT LLM TESTS (llama.cpp :8008)")
print("=" * 60)

llm_prompt_path = os.path.join(
    os.path.dirname(__file__), "..", "..", "prompts", "llm-system-prompt.md"
)
with open(llm_prompt_path, "r", encoding="utf-8") as f:
    llm_system_prompt = f.read()

llm_tests = [
    ("D1", "What is Bitcoin trading at?"),
    ("D2", "Tell me the current price of gold and provide a link"),
    ("D3", "What is the S&P 500 at today? Include a source link."),
]

llm_results = []
for tid, query in llm_tests:
    print(f"  [{tid}] {query}")
    start = time.time()
    text, wc = bh.send_llm(query, llm_system_prompt)
    text = bh.strip_thinking(text)
    wc = len(text.split())
    elapsed = time.time() - start

    dollar = has_dollar(text)
    link = has_link(text)

    if dollar:
        grade = "FAIL"
        reason = "Contains dollar amount without tool call"
    elif (
        "tavily" in text.lower()
        or "tool" in text.lower()
        or "requires" in text.lower()
        or "live data" in text.lower()
    ):
        grade = "PASS"
        reason = "Correctly refused / redirected to tool"
    else:
        grade = "WARN"
        reason = "Check manually"

    preview = text[:200].replace("\n", " ")
    print(f"         {wc}w | $={dollar} | link={link} | {grade} ({elapsed:.1f}s)")
    print(f"         >> {preview}")
    print()
    llm_results.append(
        {
            "id": tid,
            "query": query,
            "response": text,
            "word_count": wc,
            "has_dollar": dollar,
            "has_link": link,
            "grade": grade,
            "reason": reason,
            "elapsed": round(elapsed, 1),
        }
    )

llm_passed = sum(1 for r in llm_results if r["grade"] == "PASS")
print(f"  Direct LLM Score: {llm_passed}/{len(llm_results)}")
for r in llm_results:
    print(f"  [{r['id']}] {r['grade']:4s} | {r['reason']}")

# Save all results
all_results = {"workspace_tests": results, "direct_llm_tests": llm_results}
out_path = os.path.join(
    os.path.dirname(__file__), "results", "adhoc_link_check_tests.json"
)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to {out_path}")
