# Workflow Recipes

## Research Workflow (Multi-Source Synthesis)

1. **Check auto-RAG first.** Examine Context: section for relevant chunks. If it fully answers the question, cite and stop.
2. **Memory lookup.** Call search_nodes for stable facts (ports, configs, decisions). Cite: "Per Memory graph: [entity] records..."
3. **Context7 for API docs.** resolve-library-id then get-library-docs for current library documentation.
4. **Tavily for live data.** Mandatory first tool for prices, exchange rates, scores, current events. Present result with inline source link.
5. **Fetch for known URLs.** Clean markdown from specific pages. If truncated content has the answer, deliver immediately.

**Fallback chains (MCP):**
- Web content: Tavily -> Fetch -> Playwright
- Library docs: Context7 -> Tavily -> Fetch
- Local knowledge: Memory -> workspace context -> Filesystem

Cite each source as findings accumulate. Capture key findings in response text (survives compression).

## Debugging Workflow (Inspect-Modify-Verify)

**NOTE:** Desktop Commander and Filesystem are MCP tools, not AnythingLLM Agent Skills. Suggest manual checks or MCP interface when needed from AnythingLLM.

1. **Inspect:** `docker ps` to check container status (anyloom-qdrant on 6333, anyloom-llm on 8008, anyloom-embedding on 8009, anyloom-anythingllm on 3001)
2. **Diagnose:** Check logs (`docker logs [container]`), check VRAM (`nvidia-smi` — should be ~5.0-5.5 GB), check ports (`netstat -ano | findstr [port]`)
3. **Modify:** Restart containers, fix configs
4. **Verify:** Re-run checks to confirm the fix
5. **Log:** Record findings in response text. Store resolved errors in Memory.

Use Sequential Thinking (MCP) for 3+ step diagnostics. State the plan before executing.

## Memory Graph Management

### Storing facts:
1. search_nodes first — check for existing entities
2. If found: add_observations to append new facts
3. If not found: create_entities with PascalCase name, snake_case type
4. create_relations for stable connections between entities

### What to store:
- Port mappings (8008 = llama.cpp, 6333 = Qdrant, 8009 = embedding, 3001 = AnythingLLM UI)
- Configuration values that rarely change
- Resolved error patterns (symptom + root cause + fix as separate observations)
- Architecture decisions with rationale
- User preferences

### Naming conventions:
- Entity names: PascalCase (QdrantMCP, BGEm3Config, AnythingLLMWorkspace)
- Entity types: snake_case (service_config, error_resolution, port_mapping)
- Relations: snake_case (serves, depends_on, configured_in)

## DyTopo Escalation

Suggest swarm escalation when a task benefits from multiple specialist perspectives AND is complex enough to justify multi-round inference overhead.

**Good candidates:**
- Code review/architecture analysis (code domain)
- Mathematical proofs or multi-step calculations (math domain)
- Balanced multi-perspective analysis (general domain)

**Handle locally instead:**
- Single-file debugging
- Factual lookups
- Configuration verification
- Simple planning

**Example suggestion:** "This task would benefit from DyTopo's code domain swarm. To launch via MCP: swarm_start(task='...', domain='code', tau=0.45). Expect 2-4 minutes. Interactive chat will be slower during execution."

## Iteration Budget Planning

8-round cap per @agent session. Plan tool usage:
- System diagnostic (docker ps + logs + verify): 3-4 iterations
- File read + analyze: 1-2 iterations
- Web search + synthesize: 2-3 iterations
- Memory search + store: 2-3 iterations

Break complex tasks into focused sub-requests for fresh budgets.
