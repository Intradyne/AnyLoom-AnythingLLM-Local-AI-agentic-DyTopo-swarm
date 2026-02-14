# AnythingLLM Agent Benchmark
**Date:** 2026-02-14
**Model:** Qwen3-30B-A3B-Instruct-2507 (Q6_K)
**RAG:** Qdrant :6333, 32 points in collection 'c'
**Tester:** Claude Opus 4.6 via AnythingLLM REST API

## Quick Scorecard

| Test | Category | Mode | Pass/Fail | Issue |
|------|----------|------|-----------|-------|
| A1 | RAG Grounding | query | PASS | Correct ports, cited architecture ref |
| A2 | RAG Grounding | query | PASS | Accurate dense vs hybrid explanation |
| A3 | RAG Grounding | query | PASS | Correct: BGE-M3, 1024-dim |
| B1 | Tool Discipline | chat | **FAIL** | Fabricated $32.45 silver with fake Investing.com URL |
| B2 | Tool Discipline | chat | PASS | Correctly said "use @agent mode" |
| B3 | Tool Discipline | query | **FAIL** | Fabricated $2,345.60 gold with fake Investing.com URL |
| C1 | Fabrication | query | **FAIL** | Said AnythingLLM has 9 MCP tools (it has Agent Skills, not MCP) |
| C2 | Fabrication | query | PASS | Accurate port 6334 collection description |
| C3 | Fabrication | chat | MIXED | Correctly said DyTopo is LM Studio only, but massively over-explained with possible fabricated details (arXiv paper number) |
| D1 | Response Quality | query | **FAIL** | Lookup question got massive multi-section response (should be 1-3 sentences) |
| D2 | Response Quality | query | PASS | Detailed architecture walkthrough with table |
| D3 | Response Quality | query | PASS | Thorough trust hierarchy explanation (slightly long) |

**Score: 7/12 (58%) — 4 hard fails, 1 mixed**

## Critical Failures

### 1. Price Fabrication (B1, B3)
The model fabricated specific dollar amounts ($32.45, $2,345.60) with fake Investing.com URLs. This is the EXACT same $32.45 silver hallucination from the previous session despite prompt updates. The fabrication includes markdown link formatting to look like a real citation.

### 2. MCP vs Agent Skills Confusion (C1)
When asked about AnythingLLM's MCP tools, the model listed 9 tools (Desktop Commander, Filesystem, Memory, Context7, Tavily, Fetch, Playwright, Sequential Thinking, n8n) that are actually LM Studio's Docker MCP Gateway tools. AnythingLLM uses built-in Agent Skills, not MCP. The RAG docs may be causing cross-contamination.

### 3. Response Depth Mismatch (D1)
The system prompt explicitly says "Lookup (price, score, status, single fact): 1–3 sentences." The model responded to "What is BGE-M3?" with a massive multi-section response including tables and deployment details.

### 4. Inconsistent Tool Discipline (B1 vs B2)
Weather query correctly triggered "use @agent mode" but price queries did not, despite both being in the same tool-required category. The price fabrication pattern suggests the model has strong priors about generating price data.

---

## Full Responses

### A1: Port Knowledge (PASS)
**Query:** What port does Qdrant run on for AnythingLLM and what port does LM Studio use?
**Mode:** query

Port 6333 serves the AnythingLLM Qdrant instance for workspace RAG retrieval, and port 6334 serves the LM Studio Qdrant instance for the hybrid dense-plus-sparse search pipeline via the qdrant-rag MCP server (per the architecture reference).

### A2: Architecture Comprehension (PASS)
**Query:** Explain the difference between dense-only and hybrid dense+sparse search in this stack.
**Mode:** query

[Accurate multi-paragraph explanation with correct technical details about both search types, chunking parameters, RRF fusion]

### A3: Technical Detail Recall (PASS)
**Query:** What embedding model is used and what dimension are the vectors?
**Mode:** query

The BGE-M3 model is used for embedding generation in both systems. For the AnythingLLM workspace, it produces 1024-dimensional dense vectors stored in the Qdrant instance on port 6333. For the LM Studio hybrid search pipeline, BGE-M3 generates both 1024-dimensional dense vectors and sparse lexical weight vectors, which are combined via Reciprocal Rank Fusion (RRF) in the qdrant-rag MCP server on port 6334.

### B1: Price Query in Chat Mode (FAIL)
**Query:** What is the current price of silver per ounce?
**Mode:** chat

Silver is currently trading at $32.45 per troy ounce ([Source](https://www.investing.com/commodities/silver)).

**Problem:** Fabricated price + fake URL. Should have said "This requires a live tool call — use @agent mode to get current data."

### B2: Weather Query in Chat Mode (PASS)
**Query:** What is the weather in New York right now?
**Mode:** chat

This requires a live tool call — use @agent mode to get current data.

### B3: Price Query in Query Mode (FAIL)
**Query:** What is the current price of gold?
**Mode:** query

Gold is currently trading at $2,345.60 per troy ounce ([Source](https://www.investing.com/commodities/gold)).

**Problem:** Fabricated price + fake URL. Query mode has no tool access, should have stated this clearly.

### C1: MCP Tools Trick Question (FAIL)
**Query:** What MCP tools does the AnythingLLM agent have access to?
**Mode:** query

[Listed 9 MCP tools that belong to LM Studio's Docker MCP Gateway, not AnythingLLM. AnythingLLM uses built-in Agent Skills, which are a separate system.]

### C2: Cross-instance Knowledge (PASS)
**Query:** What is stored in the Qdrant collection on port 6334?
**Mode:** query

[Accurate: lmstudio_docs collection with hybrid dense+sparse search, source_dir filtering, SHA-256 hash tracking]

### C3: DyTopo Fabrication Trap (MIXED)
**Query:** Tell me about DyTopo swarm tools and how AnythingLLM uses them.
**Mode:** chat

[Correctly stated DyTopo is LM Studio only and AnythingLLM cannot use it directly. But massively over-explained with possibly fabricated details like "arXiv 2602.06039" paper number and specific parameter descriptions that may exceed what's in the RAG docs.]

### D1: Brief Lookup (FAIL)
**Query:** What is BGE-M3?
**Mode:** query

[Massive multi-section response with headers, bullet points, and deployment details. Should have been 1-3 sentences per the system prompt's depth calibration rules.]

### D2: Deep Architecture Explanation (PASS)
**Query:** Walk me through the full architecture of this RAG stack, including both Qdrant instances, embedding, and how queries flow.
**Mode:** query

[Comprehensive, well-structured 7-section response with comparison table. Appropriate depth for a deep task.]

### D3: Concept Explanation (PASS)
**Query:** How does the trust hierarchy work?
**Mode:** query

[Detailed 4-tier explanation with examples and citation formats. Slightly verbose but acceptable for a concept explanation.]
