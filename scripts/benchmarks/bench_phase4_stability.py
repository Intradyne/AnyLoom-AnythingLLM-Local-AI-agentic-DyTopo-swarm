"""Phase 4: Depth Stability - 8 queries x3 runs each on workspace c

Tests whether responses are deterministic at temp 0.1.  Queries get
progressively harder: D3-D5 are simple lookups, D6-D10 escalate from
two-fact recall through multi-axis comparison to edge-case reasoning.
Spread > 15 words across 3 runs = Stochastic.
"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = "92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3"
SLUG = "c"
bh.init(API_KEY, SLUG)

STABILITY_QUERIES = [
    # --- Original (simple lookups) ---
    ("D3",  "What is the decision cascade?"),
    ("D4",  "What is RRF?"),
    ("D5",  "What embedding model does this stack use?"),
    # --- New: progressively harder ---
    ("D6",  "What are the chunk size and overlap for AnythingLLM's RAG pipeline?"),
    ("D7",  "How does the Memory knowledge graph fit into the decision cascade?"),
    ("D8",  "What are all the MCP tools available to the LM Studio agent?"),
    ("D9",  "Compare the chunking strategies, embedding formats, and retrieval "
            "mechanisms of the two RAG pipelines side by side"),
    ("D10", "If rag_search returns zero results on port 6334, what should the "
            "agent do next according to the decision cascade, and why?"),
]

RUNS = 3

results = {}
for qid, query in STABILITY_QUERIES:
    results[qid] = []
    for run in range(RUNS):
        print(f"  {qid} run {run+1}...", flush=True)
        raw_text, raw_words, sources = bh.send(query, "query")
        text = bh.strip_thinking(raw_text)
        words = len(text.split())
        results[qid].append({"run": run + 1, "word_count": words, "text": text[:500]})
        print(f"    {qid} run {run+1}: {words} words", flush=True)
        time.sleep(3)

# Analysis
analysis = {}
for qid, _ in STABILITY_QUERIES:
    counts = [r["word_count"] for r in results[qid]]
    avg = sum(counts) / len(counts)
    spread = max(counts) - min(counts)
    verdict = "Stochastic" if spread > 15 else "Deterministic"
    analysis[qid] = {
        "min": min(counts), "max": max(counts),
        "avg": round(avg, 1), "spread": spread,
        "verdict": verdict
    }
    print(f"\n{qid}: min={min(counts)}, max={max(counts)}, "
          f"avg={avg:.0f}, spread={spread} -> {verdict}", flush=True)

det = sum(1 for a in analysis.values() if a["verdict"] == "Deterministic")
print(f"\nPhase 4: {det}/{len(analysis)} deterministic", flush=True)

output = {"runs": results, "analysis": analysis}
out = os.path.join(os.path.dirname(__file__), "results", "benchmarker_phase4_stability.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print(f"Saved to {out}", flush=True)
