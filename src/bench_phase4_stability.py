"""Phase 4: Depth Stability - rerun D3 and D4 x3 each on workspace c"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = "92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3"
SLUG = "c"
bh.init(API_KEY, SLUG)

STABILITY_QUERIES = [
    ("D3", "What is the decision cascade?"),
    ("D4", "What is RRF?"),
]

results = {}
for qid, query in STABILITY_QUERIES:
    results[qid] = []
    for run in range(3):
        print(f"  {qid} run {run+1}...", flush=True)
        text, words, sources = bh.send(query, "query")
        results[qid].append({"run": run + 1, "word_count": words, "text": text[:500]})
        print(f"    {qid} run {run+1}: {words} words", flush=True)
        time.sleep(3)

# Analysis
analysis = {}
for qid in ["D3", "D4"]:
    counts = [r["word_count"] for r in results[qid]]
    avg = sum(counts) / len(counts)
    spread = max(counts) - min(counts)
    verdict = "Stochastic" if spread > 15 else "Deterministic"
    analysis[qid] = {
        "min": min(counts), "max": max(counts),
        "avg": round(avg, 1), "spread": spread,
        "verdict": verdict
    }
    print(f"\n{qid}: min={min(counts)}, max={max(counts)}, avg={avg:.0f}, spread={spread} -> {verdict}", flush=True)

output = {"runs": results, "analysis": analysis}
out = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarker_phase4_stability.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"\nPhase 4 done. Results: {out}", flush=True)
