# Operational Troubleshooting and Maintenance Reference

Reference documentation for diagnosing common issues, maintaining the RAG index, managing Docker containers, and resolving operational problems in the Loom stack. Each section can be retrieved independently by the hybrid RAG pipeline.


## How do you verify that both Qdrant containers are running and healthy?

<!-- Verified: 2026-02-13 -->

The Loom stack runs two independent Qdrant Docker containers — port 6333 for AnythingLLM's dense-only workspace RAG and port 6334 for the MCP qdrant-rag server's hybrid search pipeline. Each operates with separate data volumes and separate configurations.

To check container status, run `docker ps` via Desktop Commander and look for two Qdrant entries with "Up" status and port mappings 6333 and 6334. If a container is missing, check `docker ps -a` for exit status, inspect logs with `docker logs <container>`, and restart with `docker start <container>`. For the port 6334 instance specifically, the rag_status tool returns endpoint URL, collection name (lmstudio_docs), point count, and staleness check. For port 6333, AnythingLLM manages health checking through its own workspace interface at localhost:3001.

Common container issues: stopped after host reboot (restart with docker start), port conflicts (check with `netstat -ano | findstr 6333`), and disk space exhaustion on Docker volumes (check with `docker system df`).

<!-- Related: Qdrant troubleshooting, Docker containers, port 6333, port 6334, container health, docker ps, docker logs, rag_status, container restart, port conflict, disk space, desktop commander -->


## How do you maintain the RAG index and detect stale documents?

<!-- Verified: 2026-02-13 -->

The qdrant-rag MCP server on port 6334 uses per-file incremental sync. Each file's SHA-256 content hash is stored in .rag_state.json. On rag_search or rag_reindex, current hashes are compared against stored state to detect additions, modifications, and deletions. Only changed files are re-embedded, making incremental updates complete in 3-10 seconds.

Call rag_status to check freshness — it reports stale (modified), new (unindexed), or deleted files. If stale files exist, run rag_reindex() for incremental sync. For a full rebuild after changing chunking configuration or suspected collection corruption, use rag_reindex(force=True) which deletes and rebuilds the collection in 30-60 seconds.

The server indexes from two source directories: "lmstudio" and "anythingllm" docs, stored in the same lmstudio_docs collection with source_dir payload filtering. Use rag_sources to list directories and file counts, and rag_file_info(filename) for per-file chunk count and hash verification. For AnythingLLM's port 6333 pipeline, document management happens through the web UI at localhost:3001 — no MCP tools manage that index.

<!-- Related: RAG index maintenance, incremental sync, rag_reindex, rag_status, rag_sources, rag_file_info, stale files, SHA-256, .rag_state.json, full rebuild, source_dir filtering, lmstudio_docs, document upload, port 3001 -->


## How do you diagnose VRAM pressure and GPU performance issues?

<!-- Verified: 2026-02-13 -->

At 80K context, the RTX 5090's 32 GB VRAM is allocated: 25.1 GB Qwen3-30B-A3B Q6_K weights, 0.6 GB BGE-M3 Q8_0 GGUF embedding model, ~1.0 GB CUDA overhead, ~3.75 GB KV cache (Q8_0 at 80K) — totaling ~30.5 GB with ~1.5 GB headroom.

Symptoms of VRAM pressure: LM Studio unresponsive or crashing, slower inference as context approaches 80K, CUDA OOM errors. Diagnose with `nvidia-smi` via Desktop Commander — usage above 31 GB indicates pressure. Mitigation: start a new conversation (resets KV cache), reduce context length in LM Studio settings (64K saves ~0.75 GB), or avoid simultaneous inference from both frontends.

The KV cache scales with context length: 32K uses ~1.5 GB, 64K ~3.0 GB, 80K ~3.75 GB, 96K ~4.5 GB (tight). The current 80K default is the practical sweet spot for Q6_K quality on 32 GB VRAM. Going beyond 96K would require switching to Q5_K_M quantization.

<!-- Related: VRAM budget, RTX 5090, 32 GB, 30.5 GB utilization, KV cache, Q8_0 quantization, CUDA out of memory, nvidia-smi, GPU monitoring, context length, VRAM pressure, headroom, Q6_K, Q5_K_M -->


## How do you troubleshoot inference latency and request queuing?

<!-- Verified: 2026-02-13 -->

Both the LM Studio agent and the AnythingLLM workspace agent share the inference endpoint at localhost port 1234. DyTopo swarm agents also call this endpoint. Requests queue sequentially — there is no parallel inference with a single GPU model.

When a DyTopo swarm is actively running (multiple agent calls per round, 3-5 rounds typical), interactive chat responses are delayed because swarm inference calls queue ahead. Check for active swarms with swarm_status if latency is unexpectedly high. Each swarm round involves descriptor-only calls (fast, 256 max_tokens, temp 0.1) followed by full work calls (4096 max_tokens, temp 0.3), plus a manager call per round.

CPU-side models load lazily: BGE-M3 (FlagEmbedding, ~2.3 GB RAM) on first rag_search, MiniLM-L6-v2 (~80 MB RAM) on first swarm. First-time BGE-M3 loading takes 30-60 seconds and may appear as a hang. Subsequent calls use the loaded singleton instantly. If the qdrant-rag server process restarts, both models must reload on next use.

<!-- Related: inference latency, request queuing, port 1234, DyTopo swarm latency, concurrent requests, LM Studio performance, model loading, BGE-M3 loading time, MiniLM lazy loading, CPU models, swarm_status, swarm overhead -->


## How to handle Fetch truncation on large or JS-heavy pages

<!-- Verified: 2026-02-13 -->

Fetch returns pages in segments. When the response ends with "Content truncated. Call the fetch tool with start_index of N..." the page content exceeds a single Fetch segment. First, check whether the truncated content already contains the answer — if yes, deliver it immediately without fetching more. If the answer genuinely requires content beyond the truncated portion, call Fetch once more with the indicated start_index. After 3 consecutive truncation-and-refetch cycles on the same URL, the page is too large or JS-heavy for Fetch. Switch to Tavily for the same query, or report what was found from partial content.

Root cause: financial data sites (investing.com, bloomberg.com, yahoo finance) and JS-heavy SPAs return massive HTML that exceeds Fetch segment limits. Call Tavily instead for market prices, exchange rates, scores, and current events — Tavily returns structured concise results in a single call.

<!-- Related: Fetch truncation, start_index, JS-heavy pages, financial sites, Tavily, 3-truncation rule, partial content, large pages -->

## How to handle tool result dates that contradict training knowledge

<!-- Verified: 2026-02-14 -->

Symptom: A tool returns data timestamped with a date the agent believes is in the future, or returns a current value that contradicts the agent's training-era knowledge of that value.

Cause: The agent's training data has an older knowledge cutoff. Tool results reflect current reality. The trust hierarchy applies to dates and timestamps: tool-verified facts rank above training knowledge.

Resolution: Trust the tool-returned date. Present it as a current fact. Do not add disclaimers suggesting the date is an error, simulation, or forecast. Do not qualify the result with "according to" or "if this data is accurate" hedging. The tool queried a live source — its timestamp is the current date and its value is the current value.

<!-- Related: tool trust, date conflict, training knowledge, trust hierarchy, timestamps, live data, knowledge cutoff, tool-verified facts -->


## Response cites a tool but data appears stale

<!-- Verified: 2026-02-14 -->

Symptom: The response says "per Tavily" or "based on live data" but the value matches training-knowledge-era data rather than current market reality. The price or fact looks stale despite the tool citation.

Cause: The agent generated an answer from training knowledge and attributed it to a tool without actually calling the tool, or called the tool but used the training-knowledge value in the response instead of the tool result. This is a hallucinated citation — it falsely claims tool-verified authority for unverified data. The worst case is fabricating both a price and a source name (e.g., inventing a financial data provider) without calling any tool — this combines hallucinated data with a hallucinated source attribution.

Resolution: For time-sensitive queries, call the tool BEFORE generating any answer text. Build the response entirely from the tool result. Every value, source name, and URL in the response must come from the tool result, not from training knowledge or from prior prompt examples. Only cite a tool when the cited data came from that tool's response in the current turn. The correct flow is: recognize time-sensitive query → call tool → read result → build answer from result with source link.

<!-- Related: hallucinated citation, stale data, tool citation, Tavily, training knowledge, anchoring, tool-call-first, citation integrity -->


## What are the common error patterns and how do you resolve them?

<!-- Verified: 2026-02-13 -->

**Connection refused on port 6334** (rag_search fails): Qdrant container is down. Run `docker ps`, then `docker start <container>`. After restart, call rag_status to verify the collection. If the container repeatedly exits, check `docker logs` for disk or config errors.

**Connection refused on port 6333** (AnythingLLM RAG stops working): Same approach — check docker ps, restart container, verify through AnythingLLM at localhost:3001.

**rag_search returns empty for a known topic**: Check rag_status for stale files, run rag_reindex(). If current, rephrase the query — change angle, adjust source filter, or modify limit. Verify the target document is indexed with rag_file_info.

**LM Studio timeout**: Check for active DyTopo swarms via swarm_status. Check VRAM with nvidia-smi. If context is very long, start a fresh conversation to reset KV cache.

**Memory entity fragmentation**: Search with search_nodes using name variations to find duplicates. Consolidate by adding observations to the canonical entity. Prevent by always calling search_nodes before create_entities.

**BGE-M3 loading exceeds 60 seconds**: Normal on first load (~2.3 GB model). If consistently over 120 seconds, check system RAM availability (94 GB DDR5 total). Subsequent searches use the loaded singleton.

<!-- Related: error patterns, connection refused, port 6333, port 6334, empty results, rag_search troubleshooting, timeout, Memory fragmentation, BGE-M3 loading, Docker restart, rag_status, rag_reindex, nvidia-smi, search_nodes, diagnostic steps -->
