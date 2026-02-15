"""Phase 6: Showcase Gallery - 6 curated queries on workspace c"""
import sys, os, json, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
import benchmark_helpers as bh

API_KEY = "92JHT3J-PMF4SGA-GT0X50Y-RMGKDT3"
SLUG = "c"
bh.init(API_KEY, SLUG)

SHOWCASE_QUERIES = [
    ("S1", "What is BGE-M3?", "query", "Perfect Lookup"),
    ("S2", "What's gold trading at right now?", "chat", "Price Fabrication Guard"),
    ("S3", "Search Memory for all entities related to Qdrant", "chat", "Tool Boundary Awareness"),
    ("S4", "Walk me through the full architecture of this stack, from user query to Qdrant retrieval and back", "query", "Deep Architecture Knowledge"),
    ("S5", "How does AnythingLLM's RAG pipeline differ from the MCP qdrant-rag server?", "query", "Explanation-Tier Comparison"),
    ("S6", "What chunking strategy does this workspace use?", "query", "RAG-Grounded Citation"),
    ("S7", "What are the memory limits for each Qdrant container?", "query", "Multi-Fact Numeric Recall"),
]

showcase = []
for sid, query, mode, label in SHOWCASE_QUERIES:
    print(f"Sending {sid} ({label})...", flush=True)
    raw_text, raw_words, sources = bh.send(query, mode)
    text = bh.strip_thinking(raw_text)
    words = len(text.split())
    showcase.append({
        "id": sid, "label": label, "query": query, "mode": mode,
        "response": text, "word_count": words
    })
    print(f"  {sid} ({label}): {words} words", flush=True)
    time.sleep(3)

out = os.path.join(os.path.dirname(__file__), "results", "benchmarker_phase6_showcase.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(showcase, f, indent=2, ensure_ascii=False)
print(f"\nPhase 6 done. Results: {out}", flush=True)
