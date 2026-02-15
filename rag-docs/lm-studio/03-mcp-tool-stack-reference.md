# MCP Tool Stack Reference

## What MCP servers are configured for the LM Studio agent?

<!-- Verified: 2026-02-13 -->

LM Studio connects to two MCP server entries defined in `C:\Users\User\.lmstudio\config\mcp.json`. The first entry is the Docker MCP Gateway (key: "MCP_DOCKER"), which provides 9 containerized tool servers: Desktop Commander for shell execution and process management, Filesystem (MCP Filesystem tool — reads, writes, searches, and lists files. NOT Windows Explorer or File Explorer.) for file read/write/search operations, Memory for a local persistent knowledge graph, Context7 for external library documentation lookup, Tavily for web search, Fetch for URL content retrieval, Playwright for browser automation, Sequential Thinking for structured reasoning scratchpad, and n8n for workflow webhooks (available but currently inactive). The second entry is the qdrant-rag server (key: "qdrant-rag"), a native Python process running `qdrant_mcp_server.py` that provides 8 tools: 5 RAG tools (`rag_search`, `rag_status`, `rag_reindex`, `rag_sources`, `rag_file_info`) and 3 DyTopo swarm tools (`swarm_start`, `swarm_status`, `swarm_result`). The swarm tools are thin MCP wrappers that delegate to the `src/dytopo/` package (8 core modules plus sub-packages for observability, safeguards, messaging, async routing, and delegation). DyTopo configuration is loaded from `dytopo_config.yaml` at the project root. The combined tool definitions occupy approximately 6,900 tokens in the context window.

<!-- Related: MCP servers, mcp.json, Docker MCP Gateway, qdrant-rag, tool list, Docker containers, LM Studio configuration -->

## How does the Docker MCP Gateway manage containerized tools?

<!-- Verified: 2026-02-13 -->

The Docker MCP Gateway runs via `docker mcp gateway run` and manages 9 containerized MCP servers with lazy lifecycle management. Containers start on first tool call and stop when idle, which means unused tools consume zero resources. The gateway is configured in mcp.json with the command `docker` and args `["mcp", "gateway", "run"]`, along with environment variables for LOCALAPPDATA, ProgramData, and ProgramFiles paths needed by the Windows host. The gateway handles routing tool calls to the appropriate container and manages container lifecycle automatically. Both LM Studio and AnythingLLM connect to the same Docker MCP Gateway through their respective mcp.json configurations, sharing access to the same set of containerized tools. This is why the Memory knowledge graph is consistent across both frontends — both agents on this machine call the same Memory container through the same gateway.

<!-- Related: Docker MCP Gateway, lazy lifecycle, container management, mcp.json, docker mcp gateway run, shared tools -->

## What RAG tools does the qdrant-rag server provide?

<!-- Verified: 2026-02-13 -->

The qdrant-rag server provides 5 RAG tools for interacting with the hybrid search pipeline on Qdrant port 6334. `rag_search(query, limit, source)` performs hybrid dense+sparse search against the `lmstudio_docs` collection — `query` is a natural language string (required), `limit` sets the result count from 1 to 10 with a default of 5, and `source` optionally filters to a specific source directory ("lmstudio" or "anythingllm") or filename. `rag_status()` takes no arguments and returns collection health, total point count, and a staleness check comparing current file hashes against the `.rag_state.json` state file. `rag_reindex(force)` triggers document re-indexing — `force=false` (default) performs incremental sync of changed files only, while `force=true` deletes and rebuilds the entire collection. `rag_sources()` returns the list of configured source directories, their filesystem paths, and file counts. `rag_file_info(filename)` returns metadata about a specific indexed file including its SHA-256 hash, chunk count, and section headers.

<!-- Related: rag_search, rag_status, rag_reindex, rag_sources, rag_file_info, RAG tools, hybrid search, lmstudio_docs -->

## What DyTopo swarm tools does the qdrant-rag server provide?

<!-- Verified: 2026-02-15 -->

The qdrant-rag server provides 3 DyTopo swarm tools that delegate to the `src/dytopo/` package (8 core modules plus sub-packages, implementing dynamic multi-agent topology routing based on arXiv 2602.06039). `swarm_start(task, domain, tau, k_in, max_rounds)` creates a `SwarmTask` Pydantic object and launches `run_swarm()` via `asyncio.create_task()` — `task` (required) describes the problem, `domain` selects the agent team ("code", "math", or "general", default "code"), `tau` sets the routing similarity threshold (default 0.3), `k_in` caps inbound connections per agent (default 3), and `max_rounds` limits iterations (default 5, maximum 10). The tool returns a `task_id` immediately. `swarm_status(task_id)` returns the current round number, LLM call count, elapsed time, progress message, and completion status. `swarm_result(task_id, include_topology)` retrieves the final output including `SwarmMetrics` (total rounds, tokens, wall time, routing density per round, convergence point, per-agent success rates and latencies) — set `include_topology=true` to see the per-round communication graph with edges, similarity scores, and topological execution order. All swarm inference runs through a backend-agnostic AsyncOpenAI client (selects LM Studio on port 1234 or vLLM on port 8000 based on the `concurrency.backend` config setting), with semaphore-controlled concurrency (max_concurrent=1 for LM Studio, 8 for vLLM), parallelized descriptor generation and tiered agent execution via `asyncio.gather()`, and tenacity retry (3 attempts, exponential backoff). DyTopo configuration is loaded from `dytopo_config.yaml` at the project root. The server stores up to 20 concurrent tasks, evicting the oldest completed tasks when the limit is reached.

<!-- Related: swarm_start, swarm_status, swarm_result, DyTopo, multi-agent, swarm tools, task_id, domain, tau, k_in, SwarmTask, SwarmMetrics, dytopo_config.yaml, src/dytopo -->

## What web research and content retrieval tools are available?

<!-- Verified: 2026-02-13 -->

Four MCP tools handle web research and external content retrieval. **Tavily** is the mandatory first tool for time-sensitive queries — call it BEFORE writing any answer text. This includes prices, exchange rates, scores, weather, current events, and any "tell me about [tradeable asset]" query. Generating an answer from training knowledge first creates an anchoring problem: the stale value persists even when the tool returns different data. The HARD STOP covers all numeric framing — "ballpark", "approximately", "roughly", "based on training data", "last you knew", "even if it's not current", and "what was it worth when..." — not just "current price." The instruction is: never output any numeric price, rate, or financial figure for any tradeable asset, regardless of how the user frames the request. Correct flow: call Tavily → read result → build answer from result. Present concisely: a price query returns one sentence with a link. Citation integrity: cite Tavily only when it was called in the current turn and the cited data came from that response. Attributing training-knowledge data to Tavily is a hallucinated citation — fabricating a source name and price without calling a tool is the worst failure mode. Prefer Tavily over Fetch for search-style queries. Financial sites truncate via Fetch — call Tavily for market data. Thread the source URL as a markdown link following this format: `[Asset] is at [tool-returned price] ([Source](tool-returned-url))` — fill the bracketed placeholders exclusively from the Tavily result, never from training data or prior examples. **Fetch** retrieves raw content from a known URL — documentation pages, API endpoints, static content. When a truncated response already contains the answer, deliver it immediately. After 3 truncation cycles, switch to Tavily. Cite inline: `per the [Qdrant docs](url), batch upsert accepts...`. **Context7** provides current API docs for external libraries via `resolve-library-id` then `get-library-docs`. Cite the library with documentation link. **Playwright** automates a browser for JS-rendered or login-protected pages — consumes 5-6 tool calls, so use only after Fetch and Tavily fail. Cite all web tool sources with inline markdown links — no "Verified via:" blocks, no separate "Source:" lines.

<!-- Related: Fetch, Tavily, Playwright, Context7, web search, URL retrieval, browser automation, library docs, API documentation, HARD STOP, ballpark, approximately, roughly, price fabrication guard, broadened framing -->

## How should Desktop Commander and Filesystem be distinguished?

<!-- Verified: 2026-02-13 -->

Desktop Commander and Filesystem (MCP Filesystem tool — reads, writes, searches, and lists files. NOT Windows Explorer or File Explorer.) serve different purposes despite both interacting with the local system. Desktop Commander handles shell command execution via `execute_command` (running scripts, `docker ps`, port checks, system state queries) and process management. Filesystem handles direct file content operations: `read_file`, `write_file`, `create_directory`, `search_files`, `list_directory`. The system_ops workflow pattern uses both tools in a multi-pass cycle: Inspect (read current state with Filesystem or Desktop Commander) → Modify (write changes with Filesystem or execute with Desktop Commander) → Verify (re-read or re-check to confirm the change took effect) → Log (record the outcome). This inspect-modify-verify pattern prevents blind writes and catches side effects. Desktop Commander runs commands in a shell and returns output; Filesystem operates directly on file contents and directory structures.

<!-- Related: Desktop Commander, Filesystem, shell commands, file operations, execute_command, read_file, write_file, search_files, list_directory -->

## How does the local Memory knowledge graph operate?

<!-- Verified: 2026-02-13 -->

The Memory MCP server maintains a local, private, persistent knowledge graph accessed by both the LM Studio and AnythingLLM agents on this machine through the Docker MCP Gateway. Memory is one of the most valuable tools available — use it eagerly to record discoveries, recall context, and build institutional knowledge across sessions. Memory supports `create_entities` for adding nodes, `create_relations` for connecting nodes, `add_observations` for appending facts to existing entities, `search_nodes` for finding entities by name or content, `read_graph` for retrieving the full graph, and `delete_entities` / `delete_relations` for removal. Entities use PascalCase names like QdrantMCP (the MCP qdrant-rag server), BGEm3Config (the BGE-M3 embedding configuration), or LmStudioApi (the LM Studio inference endpoint). Entity types and relation names use snake_case: types like `service_config`, `architecture_decision`, `error_resolution`; relations like `serves`, `depends_on`, `replaced_by`, `configured_with`. Before creating any entity, call `search_nodes` first to check if it already exists — if it does, use `add_observations` to append new facts rather than creating a duplicate. Store stable facts (port mappings, decisions, error solutions, user preferences) and skip transient context. For file contents, store the file path instead — paths are more efficient and files can always be read fresh.

<!-- Related: Memory, knowledge graph, create_entities, search_nodes, add_observations, PascalCase, snake_case, local graph -->

## What is the token budget impact of the full tool stack?

<!-- Verified: 2026-02-13 -->

Each MCP tool definition consumes approximately 300 tokens in the context window. With roughly 15 tools across the Docker MCP Gateway (9 servers providing multiple tools each, though many share definitions) and the qdrant-rag server (8 tools), the total tool definition overhead is approximately 6,900 tokens. Combined with the system prompt at approximately 2,000 tokens, the fixed overhead is approximately 8,900 tokens. From the 80,000-token total context window, this leaves approximately 71,000 tokens for conversation history, RAG search results, tool output, and model responses. You have plenty of context room. Focus on answer quality, not token counting. For reference, `rag_search` results should be kept targeted (limit 3-5 for precise queries) to keep results focused. Broad surveys at limit 8-10 are useful when comprehensive coverage is genuinely needed. Summarizing tool results into 1-2 sentences before the next call helps preserve context as the RollingWindow context manager drops oldest messages when the window fills.

<!-- Related: token budget, tool definitions, overhead, context window, 80K tokens, 71K available, RollingWindow, context management -->
