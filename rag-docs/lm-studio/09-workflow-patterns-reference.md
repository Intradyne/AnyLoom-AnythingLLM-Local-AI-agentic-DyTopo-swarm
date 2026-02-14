# Workflow Patterns Reference

## How to investigate a multi-source architecture question

<!-- Verified: 2026-02-13 -->

When a user asks a question spanning multiple components — such as "How does data flow from user query to RAG results?" — the answer lives across several documents and possibly Memory entities. Use `/think` to plan the approach, then work from broad to specific.

Start with a broad `rag_search` to identify which documents are relevant. Read the results and identify gaps. Then use source-filtered queries to pull precise details from specific documents. Cross-check factual details (ports, collection names, service endpoints) against Memory entities for confirmation.

```
→ /think — multi-component question, need to plan the query sequence
→ rag_search(query="how does a user query reach RAG search results", limit=5)
   Observation: doc 01 describes topology, doc 02 describes hybrid search mechanics
→ rag_search(query="how does qdrant_mcp_server process a rag_search call", source="02-qdrant-rag-reference.md", limit=3)
   Observation: query encoded to dense+sparse, parallel search, RRF fusion, min score 0.005
→ search_nodes("QdrantMCP")
   Observation: Memory confirms port 6334, lmstudio_docs collection
→ Synthesize: "When you call rag_search, the query goes to qdrant_mcp_server.py on port 6334.
   BGE-M3 encodes it into dense (1024-dim) and sparse vectors on CPU.
   Both run parallel searches against the lmstudio_docs collection.
   RRF fusion merges the ranked lists. Results above 0.005 score are returned.
   Per rag_search on 01-architecture-reference and 02-qdrant-rag-reference,
   confirmed via Memory entity QdrantMCP."
```

Key principle: Start broad with `rag_search`, narrow with source filters, cross-check with Memory, synthesize across sources. Cite every source that contributed to the answer.

<!-- Related: multi-source, architecture question, broad-to-narrow, source filter, cross-check, Memory, rag_search, synthesis, citation -->

## How to debug a failing service step by step

<!-- Verified: 2026-02-13 -->

When a user reports something broken — "rag_search returns empty results" or "the swarm hangs" — use `/think` for root-cause analysis. Always check live system state before consulting documentation. Follow the dependency chain from the visible symptom back to the root cause.

The debugging workflow is: confirm the failure → check infrastructure → diagnose the root cause → apply the fix → verify the fix worked. Each step produces observations that guide the next step.

```
→ /think — debugging, need structured root-cause analysis
→ rag_status()
   Observation: collection unhealthy or 0 points — confirms the failure is real
→ Desktop Commander: execute_command("docker ps --filter name=qdrant")
   Observation: both containers running, port 6334 mapped correctly
→ Desktop Commander: execute_command("curl -s http://localhost:6334/collections/lmstudio_docs")
   Observation: collection exists but 0 points — index was wiped
→ rag_reindex(force=true)
   Observation: re-indexed 7 files, 44 points total
→ rag_status()
   Observation: healthy, 44 points, 0 stale files
→ "The lmstudio_docs collection was empty — likely wiped by a container restart.
   Confirmed Qdrant is running on port 6334 via docker ps.
   Forced full reindex: now 44 points across 7 files.
   Per rag_status (before: 0 points, after: 44 points) and Desktop Commander (docker ps)."
```

Key principle: Check live state first (docker ps, rag_status), then diagnose, then fix, then verify the fix. Each observation drives the next action. Report what changed between the before and after states.

<!-- Related: debugging, root-cause analysis, docker ps, rag_status, rag_reindex, dependency chain, verify fix, /think -->

## How to launch, monitor, and retrieve a DyTopo swarm

<!-- Verified: 2026-02-13 -->

When a user requests something benefiting from multiple specialist perspectives — code review, architecture analysis, proof verification — consider a DyTopo swarm. Use `/think` to assess whether the task genuinely benefits from collaborative multi-agent work. Simple lookups and single-tool tasks should be handled directly.

The swarm lifecycle is: assess → launch → poll → retrieve → present. Choose the domain matching the task (code for software, math for proofs/calculations, general for open analysis). Poll every 15-30 seconds — the swarm runs asynchronously.

```
→ /think — assess whether swarm is justified
   This is a code review with security implications — Developer, Tester, and Researcher
   roles will each catch different vulnerability classes. Swarm is appropriate.
→ swarm_start(task="Review for SQL injection, XSS, and auth bypass: [paste code]", domain="code")
   Observation: task_id "review-42" returned immediately
→ (wait 15-20 seconds)
→ swarm_status("review-42")
   Observation: round 2/5, Developer active, 34s elapsed
→ (wait 20 seconds)
→ swarm_status("review-42")
   Observation: round 4/5, Manager active, 72s elapsed
→ (wait 15 seconds)
→ swarm_status("review-42")
   Observation: completed, 4 rounds, 85s total
→ swarm_result("review-42")
   Observation: Developer found unsanitized input at line 23, Researcher cited OWASP A03:2021,
   Tester confirmed the injection path with a test payload, Designer suggested parameterized queries
→ "DyTopo code swarm completed in 4 rounds (85s):
   - Developer: unsanitized input on line 23 in handle_query()
   - Tester: confirmed SQL injection via payload ' OR 1=1--
   - Researcher: matches OWASP A03:2021 (Injection)
   - Designer: recommended parameterized queries with cursor.execute(sql, params)
   Per swarm_result on task review-42."
```

Key principle: Launch → poll every 15-30s → retrieve on completion → cite each agent's contribution. Present findings organized by agent role so the user sees which specialist caught what.

<!-- Related: DyTopo, swarm_start, swarm_status, swarm_result, code domain, polling, launch workflow, security review, multi-agent -->

## How to recover from tool failures using fallback chains

<!-- Verified: 2026-02-13 -->

When a tool call returns an error, empty result, or unexpected format, classify the error type first and then follow the appropriate fallback chain. The goal is silent recovery — the user sees the answer, not the failed attempts, unless all fallbacks are exhausted.

Three fallback chains cover most failures: Web content (Tavily → Fetch → Playwright), Library docs (Context7 → Tavily → Fetch), Local knowledge (Memory → rag_search → Filesystem → Desktop Commander). After two failures on the same path, change the approach entirely rather than continuing to retry.

```
Example 1 — Empty Context7 result:
→ Context7: resolve-library-id("FlagEmbedding") — resolved to "BAAI/FlagEmbedding"
→ Context7: get-library-docs(id, topic="return_sparse parameter") — empty result
   Classify: empty results on external docs → follow Library docs fallback chain → try Tavily
→ Tavily("FlagEmbedding return_sparse parameter usage")
   Observation: found GitHub README with usage example
→ "FlagEmbedding supports return_sparse=True to generate lexical weight vectors alongside dense embeddings.
   Per Tavily result from BAAI/FlagEmbedding README.
   Context7 had no coverage for this parameter."

Example 2 — Qdrant connection refused:
→ rag_search(query="port assignments") — connection refused on port 6334
   Classify: Qdrant unreachable → check container status with Desktop Commander
→ Desktop Commander: execute_command("docker ps --filter name=lmstudio-qdrant")
   Observation: container not running
→ "Qdrant on port 6334 is down — the lmstudio-qdrant container is not running.
   Confirmed via Desktop Commander: docker ps shows no matching container.
   Start it with: docker start lmstudio-qdrant"
```

Key principle: Classify the error → follow the matching fallback chain → report what was checked and found. Two failures on the same path = change approach entirely. Report failures transparently only after exhausting alternatives.

<!-- Related: error recovery, fallback chains, Context7, Tavily, Fetch, Playwright, connection refused, silent recovery, classify-then-act -->

## How to reformulate a RAG query when first results are thin

<!-- Verified: 2026-02-13 -->

When `rag_search` returns irrelevant, too-broad, or empty results, diagnose why and reformulate with the opposite strategy. Three strategies: **narrow** (add source filter, specific terms), **broaden** (drop filter, increase limit, general terms), **re-angle** (change question perspective entirely).

```
Strategy 1 — Narrow with source filter:
→ rag_search(query="how does the embedding model work", limit=5)
   Observation: returns chunks from docs 01, 02, 05, 07 — too scattered, no focused answer
→ rag_search(query="BGE-M3 FlagEmbedding CPU pipeline for hybrid search", source="05-model-embedding-reference.md", limit=3)
   Observation: precise — FlagEmbedding, dense+sparse, 2.3 GB RAM, 8 CPU threads
→ Use the precise result. Cite source: "per rag_search on 05-model-embedding-reference."

Strategy 2 — Broaden after empty results:
→ rag_search(query="how does ColBERT late interaction work in this stack", limit=3)
   Observation: 0 results — ColBERT is disabled, barely mentioned
→ rag_search(query="BGE-M3 embedding configuration options disabled features", limit=5)
   Observation: doc 05 mentions ColBERT disabled due to 50x storage overhead
→ "ColBERT late-interaction is disabled in this stack due to ~50x storage overhead
   with marginal retrieval benefit. Per rag_search on 05-model-embedding-reference."

Strategy 3 — Change question angle:
→ rag_search(query="what port does Qdrant use", limit=3)
   Observation: returns both port 6333 and 6334 chunks — ambiguous
→ rag_search(query="which Qdrant container serves the MCP hybrid search pipeline", limit=3)
   Observation: precise — port 6334, lmstudio_docs collection, RRF fusion
→ Use the precise result.
```

Key principle: Thin results → diagnose (too broad? wrong vocabulary? missing filter?) → reformulate with the opposite strategy. Each reformulation must be substantively different. Hybrid search works best with full sentences including both technical terms and plain-language descriptions.

<!-- Related: query reformulation, rag_search, source filter, narrow, broaden, re-angle, hybrid search, vocabulary, retrieval quality -->

## How to integrate Memory reads and writes into a workflow

<!-- Verified: 2026-02-13 -->

The Memory knowledge graph stores stable facts that persist across sessions. Core discipline: always `search_nodes` before creating entities. If found, use `add_observations` to append. Follow PascalCase entity names, snake_case types and relations.

```
Record a decision: User says "switching AnythingLLM temp from 0.1 to 0.2"
→ search_nodes("AnythingLLM temperature") → found AnythingLlmConfig
→ add_observations("AnythingLlmConfig", ["temperature changed to 0.2, was 0.1"])
→ "Recorded: AnythingLlmConfig temperature updated to 0.2."

Store an error solution: rag_search fails on 0-point collection
→ search_nodes("rag_search empty") → no entity
→ create_entities([{name: "RagSearchEmptyCollection", type: "error_resolution",
   observations: ["fails on 0 points", "Fix: rag_reindex(force=true)"]}])
→ create_relations([{from: "RagSearchEmptyCollection", to: "QdrantMCP", type: "affects"}])

Session start recall:
→ search_nodes("recent changes") → DyTopo tau=0.25, AnythingLLM temp=0.2
```

Key principle: `search_nodes` → `add_observations` if exists → `create_entities` only if new → `create_relations` to link. Both agents share this graph — consistency matters.

<!-- Related: Memory, search_nodes, add_observations, create_entities, create_relations, search-before-create, PascalCase, snake_case, shared graph -->

## How to use Context7 for external library documentation

<!-- Verified: 2026-02-13 -->

Context7 provides current API documentation for external libraries that may be newer than training data. The two-step process is required: first `resolve-library-id` to get the Context7 identifier, then `get-library-docs` with the resolved ID and a specific topic.

Use Context7 when the question is about a third-party library API, method signature, configuration parameter, or version-specific behavior. It sits at priority 3 in the tool ladder — after `rag_search` (for internal docs) and Memory (for stable facts), but before web search tools. If Context7 returns empty results, fall back to Tavily then Fetch.

```
→ User: "What parameters does FlagEmbedding's BGEM3FlagModel.encode() accept?"
→ Context7: resolve-library-id("FlagEmbedding")
   Observation: resolved to "BAAI/FlagEmbedding" with Context7 ID
→ Context7: get-library-docs(id, topic="BGEM3FlagModel encode parameters")
   Observation: encode() accepts sentences, batch_size, max_length, return_dense, return_sparse, return_colbert_vecs
→ "Per Context7 docs for BAAI/FlagEmbedding: BGEM3FlagModel.encode() accepts:
   - sentences: list of strings
   - batch_size: int (default 12)
   - max_length: int (default 8192)
   - return_dense: bool (default True)
   - return_sparse: bool (default False)
   - return_colbert_vecs: bool (default False)
   This stack uses return_sparse=True and max_length=1024."
```

When Context7 returns partial results, supplement with the fallback chain: Tavily for web search or Fetch for a known documentation URL. Always cite "per Context7 docs for [library]" when using Context7 results, and note which tool provided supplementary information.

<!-- Related: Context7, resolve-library-id, get-library-docs, external docs, library API, two-step process, fallback, Tavily, version-sensitive -->

## How to use Sequential Thinking for complex planning

<!-- Verified: 2026-02-13 -->

Sequential Thinking provides a visible, structured reasoning scratchpad that persists in context and can be referenced later. It differs from `/think` in important ways: `/think` produces internal chain-of-thought reasoning that the user does not see, while Sequential Thinking creates explicit structured output that both the agent and user can reference.

Use Sequential Thinking when the plan itself is the deliverable — migration plans, refactoring strategies, multi-file change sequences — or when 5+ interdependent steps need ordering before execution begins. For simpler reasoning (2-3 steps, single-file changes), `/think` is sufficient and faster.

```
→ User: "Plan the migration from single-source to multi-source RAG indexing"
→ /think — this needs structured deliberation, not just internal reasoning
→ Sequential Thinking: create thought "Migration plan: single-source to multi-source"
   Step 1: Audit current indexing code in qdrant_mcp_server.py — identify hardcoded paths
   Step 2: Design source_dir payload field and per-source hash tracking in .rag_state.json
   Step 3: Update chunk_document() to accept source_label parameter
   Step 4: Add source_dir payload index for filtered rag_search queries
   Step 5: Modify rag_search to accept optional source filter parameter
   Step 6: Update rag_reindex to iterate over multiple source directories
   Step 7: Test — index both dirs, verify filtered and unfiltered search return correct results
→ "Here's the migration plan from Sequential Thinking:
   [presents the 7-step plan with specific files and functions to modify]
   Steps 1-2 are read-only investigation. Steps 3-6 modify qdrant_mcp_server.py.
   Step 7 validates the full pipeline. Starting with step 1 now."
```

Key principle: Use Sequential Thinking when the plan itself is valuable output or when 5+ interdependent steps need explicit ordering. Use `/think` for internal reasoning that doesn't need to be presented as a structured artifact.

<!-- Related: Sequential Thinking, /think, reasoning scratchpad, planning, migration, multi-step, structured output, complex analysis -->

## How to handle cold start and orient the agent at session start

<!-- Verified: 2026-02-13 -->

At the beginning of a new session, the agent has no conversation history. Memory, RAG status, and service health must all be confirmed before acting on any user request. The cold start protocol runs three checks in sequence, then reports findings before proceeding.

This orientation matters because between sessions, containers may restart (wiping the Qdrant index), documents may be updated (making the index stale), and user preferences may have been recorded in Memory that affect how the agent should behave. Discovering these issues proactively is better than encountering them mid-task.

```
→ (session starts, user sends first message)
→ rag_status()
   Observation: lmstudio_docs healthy, 44 points, 0 stale files
→ search_nodes("recent")
   Observation: DyTopoSwarm tau=0.25, AnythingLlmConfig temp=0.2,
   last session resolved rag_search empty collection issue
→ Desktop Commander: execute_command("docker ps --format '{{.Names}} {{.Status}}'")
   Observation: anythingllm-qdrant Up 3 days, lmstudio-qdrant Up 3 days
→ "Session oriented:
   - RAG index healthy: 44 points, 0 stale (per rag_status)
   - Memory recalls: tau default now 0.25, ALM temp now 0.2
   - Both Qdrant containers running (per Desktop Commander)
   Ready. How can I help?"
```

If orientation reveals issues — stale index, missing containers, graph inconsistencies — report them before addressing the user's request. Fix what can be fixed automatically (e.g., `rag_reindex` for stale files) and report what requires user action (e.g., a stopped container).

<!-- Related: cold start, session start, orientation, rag_status, search_nodes, docker ps, service health, proactive, Memory recall -->

## How to report coverage gaps when all sources are exhausted

<!-- Verified: 2026-02-13 -->

When none of the available knowledge sources can answer a question, transparency builds trust. State exactly which sources were checked, what query was used for each, and what each returned. Then suggest concrete next steps the user could take to find the answer.

The full escalation path for a knowledge question is: `rag_search` (internal docs) → Memory `search_nodes` (stable facts) → Context7 (external library docs) → Tavily (web search) → Fetch (specific URL). If all five return empty or irrelevant results, report the gap honestly.

```
→ User: "What is the maximum number of named vectors Qdrant supports per collection?"
→ rag_search(query="maximum named vectors per Qdrant collection", limit=5)
   Observation: docs mention named vectors for dense and sparse, but no max limit stated
→ search_nodes("Qdrant named vectors limit")
   Observation: no matching entities
→ Context7: resolve-library-id("qdrant") → get-library-docs(id, topic="named vectors limit")
   Observation: docs describe named vectors but no explicit limit
→ Tavily("qdrant maximum named vectors per collection limit")
   Observation: no definitive answer in top results
→ "I checked four sources and none documents an explicit limit:
   - rag_search: describes our 2 named vectors but no max (per 02-qdrant-rag-reference)
   - Memory: no matching entities
   - Context7 Qdrant docs: describes named vector configuration but no stated limit
   - Tavily web search: no definitive answer
   This may require checking the Qdrant source code or asking in their Discord.
   The practical answer for this stack is 2 named vectors (dense + sparse), which works without issue."
```

Key principle: Check all sources in the fallback chain → state each source and what it returned → suggest actionable next steps if the gap remains. Never fabricate an answer to fill a gap.

<!-- Related: coverage gaps, exhausted sources, transparency, fallback chain, rag_search, Memory, Context7, Tavily, honest reporting -->

## How to retrieve live data (prices, scores, weather, news)

<!-- Verified: 2026-02-13 -->

Call Tavily with a concise factual query targeting the specific data point. Tavily returns structured results with source URLs and resolves real-time factual queries in a single tool call. Financial data sites (investing.com, bloomberg.com, yahoo finance) return massive JS-heavy HTML that truncates repeatedly via Fetch — call Tavily instead for any market data, sports scores, weather, or current events. Present the answer in 1-3 sentences with an inline source link.

```
→ User: "What's the current price of silver?"
→ Tavily("current silver price per ounce")
→ "Silver is trading at $32.45/oz ([Kitco](https://www.kitco.com/silver-price-today-usa/))."
```

A price check is a lookup — one sentence with a link. Providing per-gram conversions, market analysis, or technical indicators for a price lookup is a quality failure.

Correct flow: (1) recognize query involves time-sensitive data, (2) call Tavily, (3) read the result, (4) build answer from tool result with inline source link. Do not generate an answer from training knowledge and then call a tool to verify — the initial answer anchors the response and the tool result gets ignored or fitted around it.

Key principle: Real-time data = Tavily first, one call, build answer from result. Fetch is for known static URLs with clean HTML, not live data queries.

<!-- Related: live data, Tavily, real-time prices, scores, weather, news, financial data, one-call resolution, market data -->

## How to research web content with escalating tools

<!-- Verified: 2026-02-13 -->

Three web tools at different latency costs. **Tavily** resolves real-time factual queries (prices, scores, weather, news) in a single call — call it first for any search-style query. **Fetch** retrieves raw content from a known URL — call it for documentation pages and static content. If Fetch returns truncated content and the answer is already present, deliver it immediately; after 3 truncation cycles, switch to Tavily. **Playwright** automates a browser — consumes 5-6 tool calls, so call it only when Tavily and Fetch both fail.

```
Real-time data → Tavily:
→ Tavily("current silver price per ounce") → $32.45/oz from Kitco
→ "Silver is at $32.45/oz ([Kitco](https://www.kitco.com/silver-price-today-usa/))." One call.

Known URL → Fetch:
→ Fetch("https://qdrant.tech/documentation/concepts/hybrid-queries/")
→ Full content. Cite: "Per the [Qdrant hybrid queries docs](https://qdrant.tech/documentation/concepts/hybrid-queries/), RRF uses 1/(k+rank)."

JS-rendered → escalate to Playwright:
→ Fetch(huggingface_url) → minimal content (needs JS)
→ Playwright: navigate, wait, extract → full results. Cite the page URL inline.
```

Key principle: Tavily → Fetch → Playwright. Financial sites return JS-heavy HTML that truncates via Fetch — call Tavily for market data. Cite source URLs inline.

<!-- Related: web research, Fetch, Tavily, Playwright, URL citation, escalation, JavaScript rendering, web search, known URL -->

## How to safely read, modify, and verify a file

<!-- Verified: 2026-02-13 -->

The inspect-before-modifying rule requires reading a file before editing it, understanding the surrounding context, making a targeted change, and then re-reading to confirm the change took effect. Never write blind — the file's current state may differ from what you expect.

This three-step pattern (read → modify → verify) prevents several common problems: overwriting uncommitted changes, modifying the wrong line, missing side effects in surrounding code, and failing to propagate related changes.

```
→ User: "Change the RAG_CPU_THREADS default from 8 to 12 in the server"
→ Filesystem: read_file("C:\Users\User\Qdrant-RAG+Agents\src\qdrant_mcp_server.py")
   Observation: line 47: RAG_CPU_THREADS = int(os.environ.get("RAG_CPU_THREADS", "8"))
   Also note: OMP_NUM_THREADS and MKL_NUM_THREADS set to same value at lines 48-49
→ /think — changing CPU threads affects OMP and MKL too, should I update all three?
   Decision: the env vars derive from RAG_CPU_THREADS at runtime, so changing the default propagates.
   Only need to change line 47.
→ Filesystem: write_file — update line 47: "8" → "12"
→ Filesystem: read_file — confirmed line 47 now reads "12"
→ "Updated RAG_CPU_THREADS default from 8 to 12 in qdrant_mcp_server.py line 47.
   OMP_NUM_THREADS and MKL_NUM_THREADS derive from this at runtime, so they'll also use 12.
   Confirmed via re-read. Note: restart the MCP server for this to take effect."
```

Key principle: read_file → understand context and side effects → write the targeted change → read_file again to verify → report what changed, what propagates, and any action needed (like a service restart). Use `/think` when the surrounding code introduces dependencies.

<!-- Related: file editing, read-modify-verify, inspect before modifying, Filesystem, read_file, write_file, side effects, targeted change -->

## Response depth matches question complexity (hard constraint)

<!-- Verified: 2026-02-14 -->

This is a hard constraint applied to every response:

- **Lookup** (price, score, status, single fact): 1-3 sentences. Answer with source link. Stop.
- **Explanation** (how-to, comparison, concept): 1-2 short paragraphs with relevant context.
- **Complex task** (debugging, architecture, multi-step research): full structured response with evidence.

Volunteering unrequested depth is a quality failure. A user who asks "What port does Qdrant use?" wants `"Port 6334 for MCP hybrid search, port 6333 for AnythingLLM (per rag_search on 01-architecture-reference)."` — not a full architecture overview. End with the answer. Do not append "Let me know if you'd like..." or "Would you like me to..." offers.

<!-- Related: output proportionality, response depth, question complexity, concise answers, hard constraint, quality failure, no follow-up offers -->
