# AnythingLLM Agent Benchmark

> **Engine:** llama.cpp with Qwen3-30B-A3B Q4_K_M GGUF, 131K context, RTX 5090.
> Phases 1-4 and 6 run via AnythingLLM REST API; Phase 5 validated on llama.cpp directly.

**Date:** 2026-02-17
**Model:** Qwen3-30B-A3B-Instruct-2507 (Q4_K_M GGUF, ~18.6 GiB)
**RAG:** Qdrant :6333, hybrid dense+sparse
**Tester:** Python `requests` via `bench_run_all.py` (P1-P4, P6) and direct llama.cpp API (P5)

## Quick Scorecard

| Phase | Category | Score | Notes |
|-------|----------|-------|-------|
| P1 | Explanation Tier | 0/5 | All over 150w limit (275-378w); content accurate |
| P2 | Adversarial Fabrication | 5/5 | All price queries refused with identical 16w template |
| P3 | Cross-Workspace Parity | 5/5 | All pass including W5 at 145w |
| P4 | Depth Stability | 8/8 | All deterministic (spread=0 across 3 runs) |
| P5 | LLM Direct Validation | 5/5 | All pass — L5 correctly routes to Tavily |
| P6 | Showcase Gallery | 7/7 | All collected |

**Combined graded score: 15/20 (75%)**

## Key Findings

- **Fabrication guard**: Bulletproof — 10/10 price queries (direct + adversarial) refused
- **Explanation tier**: 0/5 due to verbosity (275-378w vs 150w cap). Content quality is high.
- **Depth stability**: Perfect determinism — all 8 queries produce identical output across 3 runs (spread=0)
- **LLM direct**: 5/5 — all depth calibration and tool boundary checks pass
- **Tool boundary**: S3 tool-call leakage persists (outputs `search_nodes("Qdrant")` as text)

## Score History

| Date | Score | Engine | Notes |
|------|-------|--------|-------|
| 2026-02-17 | 15/20 (75%) | llama.cpp + Blackwell image | P1 verbose (0/5), P2-P5 perfect |
| 2026-02-16 | 18/20 (90%) | llama.cpp + Blackwell image | P1 4/5, P5 L5 rag_search routing |
| 2026-02-14 | 7/12 (58%) | vLLM FP8/Q6_K | Price fabrication, MCP confusion |

---

*Full results: [benchmark-results-showcase.md](benchmark-results-showcase.md)*
*Benchmark spec: [../benchmarker.md](../benchmarker.md)*
