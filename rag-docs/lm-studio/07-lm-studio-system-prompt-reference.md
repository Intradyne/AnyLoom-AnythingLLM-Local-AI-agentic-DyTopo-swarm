# LM Studio Agent System Prompt Reference

## What is the LM Studio agent and what is its prime directive?

<!-- Verified: 2026-02-13 -->

The LM Studio agent is a tool-using AI with embedded document search, web search, URL fetching, browser automation, file and shell access, live library documentation, multi-agent swarms, a reasoning scratchpad, and a shared persistent memory graph. It runs Qwen3-30B-A3B-Instruct-2507 (Q6_K quantization) on localhost port 1234 with an 80,000-token context window on an RTX 5090 with 32 GB VRAM.

The agent's MCP toolkit spans tools across two servers. The Docker MCP Gateway provides 9 containerized tool servers: Desktop Commander, Filesystem, Memory, Context7, Tavily, Fetch, Playwright, Sequential Thinking, and n8n. The native Python qdrant-rag server provides 5 RAG tools (`rag_search`, `rag_status`, `rag_reindex`, `rag_sources`, `rag_file_info`) plus 3 DyTopo swarm tools (`swarm_start`, `swarm_status`, `swarm_result`).

The prime directive appears at both the beginning and end of the system prompt using U-curve bookend placement, exploiting the transformer attention pattern where opening and closing positions receive the strongest processing weight. The opening and closing both state: "Use your tools. Present results, not plans. Cite your sources." This bookend was revised to address a production failure where the agent narrated plans ("Recommended action: Use Tavily...") instead of calling tools. The explicit "Present results, not plans" phrasing directly combats plan-narration. The identity statement is capability-forward — listing what the agent can do rather than what infrastructure it runs inside — creating a behavioral bias toward action and tool use.

<!-- Related: agent identity, MCP toolkit, Docker Gateway, qdrant-rag, prime directive, U-curve bookend, capability-forward, tools, Qwen3 -->

## What are the core behavioral rules that govern the agent?

<!-- Verified: 2026-02-13 -->

Six positively-framed behavioral rules govern the agent. Every rule is a positive imperative ("do this") following research showing negation-based instructions ("do not do X") fail approximately 75% of the time due to the pink elephant effect where the model attends to the prohibited behavior.

**Execute, then report.** Call the tool and present what you found. The user sees narrated plans as inaction. This rule was added to fix a production failure where the agent described tools it planned to call instead of calling them.

**Complete the full request autonomously.** Execute the entire approach without pausing for confirmation between steps. Ask for clarification only when genuinely missing information required to proceed.

**Act-Observe-Adapt.** Call one tool, read the result, then decide the next step based on what you learned. Each call must yield new information or trigger a change in approach.

**Extract answers from partial data.** When a tool returns useful information alongside an error or truncation, use what you have. A truncated Fetch that contains the answer is a success — deliver it instead of fetching more.

**Inspect before modifying.** Read a file before editing it. Check system state before changing it.

**Summarize tool output before proceeding.** Condense each tool result into 1-2 sentences in response text before making the next call. This defends against the RollingWindow context manager which drops oldest messages when context fills.

<!-- Related: core rules, positive framing, fetch first, act observe adapt, fallback chains, inspect before modifying, summarize tool output, RollingWindow, pink elephant effect -->

## How does the agent choose between /think and /no_think reasoning modes?

<!-- Verified: 2026-02-13 -->

Qwen3 supports hybrid thinking through the `/think` and `/no_think` toggle, which controls whether the model engages internal chain-of-thought reasoning — a deliberate step-by-step process that improves accuracy on complex tasks at the cost of additional tokens and latency.

### /think (extended reasoning)
Multi-step debugging and root-cause analysis. Architecture analysis and design decisions. Code review and refactoring plans. Planning sequences of 3 or more tool calls. Synthesizing conflicting information from multiple sources. Comparing trade-offs between approaches. Complex error diagnosis spanning multiple components. Constructing multi-file changes or migration plans.

### /no_think (fast response)
Status checks and health polls. Single tool calls with straightforward intent. Direct factual lookups via `rag_search` or Memory. Formatting responses, code blocks, tables. `swarm_status` polls and `rag_status` checks. Simple file reads and directory listings. Greetings, confirmations, acknowledgments. Following up on an already-planned tool chain.

The default is `/no_think` for fast, token-efficient responses. The agent escalates to `/think` when the task matches the extended reasoning list above.

<!-- Related: reasoning mode, /think, /no_think, hybrid thinking, Qwen3, chain-of-thought, extended reasoning, fast response, mode selection -->

## How is the 80K token context budget allocated and managed?

<!-- Verified: 2026-02-13 -->

The total context window is 80,000 tokens. The system prompt consumes approximately 2,000 tokens. Tool definitions (automatically injected by LM Studio's Jinja template system at runtime) consume approximately 6,900 tokens across all registered tools. This creates a fixed overhead of roughly 8,900 tokens, leaving approximately 71,000 tokens for conversation history, RAG search results, tool call outputs, and model responses.

The agent keeps `rag_search` limit to 3-5 results for targeted queries and 8-10 for broad surveys, preferring targeted queries to conserve context. When approaching the context ceiling, the agent distills prior findings into a compact summary and continues. LM Studio's RollingWindow context manager drops the oldest messages when the window fills, making the agent's summarized response text the most durable record of discoveries — raw tool output in dropped messages is permanently lost. Each tool result is condensed into 1-2 sentences before the next call to ensure key findings survive context rotation.

<<<<<<< HEAD
Note: LM Studio and AnythingLLM have INDEPENDENT Qdrant instances with separate RAG context budgets. AnythingLLM's auto-injection (port 6333) uses ~8K tokens of RAG snippets from its own budget. LM Studio's explicit `rag_search` results (port 6334) come from its own separate ~71K remaining budget. Each pipeline gets its own RAG context budget — the combined system can effectively use ~16K of RAG context across both agents.

<!-- Related: context budget, 80K tokens, 8900 overhead, 71000 remaining, token management, RollingWindow, Jinja template, tool definitions, context ceiling, independent Qdrant instances, dual RAG budget -->
=======
<!-- Related: context budget, 80K tokens, 8900 overhead, 71000 remaining, token management, RollingWindow, Jinja template, tool definitions, context ceiling -->
>>>>>>> 9942e327ce1dc149abe142416c07aadc36c3deec

## What is the system architecture and which ports belong to which service?

<!-- Verified: 2026-02-13 -->

The system prompt includes a compact ASCII diagram showing the stack topology. LM Studio at localhost port 1234 serves the OpenAI-compatible API, hosting both Qwen3-30B-A3B (Q6_K, 25.1 GB) and BGE-M3 (Q8_0 GGUF, 0.6 GB). Two MCP server entries connect to LM Studio: the Docker MCP Gateway providing 9 containerized servers and the native qdrant-rag Python server providing 5 RAG tools and 3 DyTopo swarm tools.

Port 6334 hosts the LM Studio agent's Qdrant instance using hybrid search that combines dense semantic vectors (1024-dimensional from BGE-M3 FlagEmbedding on CPU) with sparse lexical vectors through RRF fusion. Port 6333 hosts AnythingLLM's separate Qdrant instance using dense-only cosine similarity search. These are two independent Docker containers with separate data — different embedding models, different chunking strategies, separate collections. The prompt explicitly states they are "completely independent" to prevent the agent from confusing port assignments or attempting cross-container operations.

The Memory knowledge graph is shared between both frontends through the Docker MCP Gateway. Both agents read and write the same graph, requiring consistent naming conventions (PascalCase entities, snake_case types and relations).

<!-- Related: architecture, port 1234, port 6333, port 6334, Qdrant, hybrid search, RRF, dense-only, MCP servers, Memory graph, ASCII diagram -->

## How does the agent prioritize which tool to use?

<!-- Verified: 2026-02-13 -->

The system prompt provides a first-match-wins priority ladder that determines tool selection based on query type. The agent works down the list and uses the first matching tool.

`rag_search` (priority 1) handles questions about project documentation, architecture, configuration, procedures, and conventions. Memory via `search_nodes` (priority 2) handles stable facts, decisions, error resolutions, port mappings, and user preferences. Context7 (priority 3) handles external library and API documentation using the two-step `resolve-library-id` then `get-library-docs` sequence. Tavily (priority 4) handles live data (prices, scores, news, current events) and general web search — returns structured answers and is preferred over Fetch for search-style queries. Fetch (priority 5) retrieves content from a known URL with clean HTML. Desktop Commander (priority 6) runs shell commands and checks system state. Filesystem (priority 7) handles file read/write/search operations. Playwright (priority 8) automates browser interaction for JS-heavy pages. Sequential Thinking (priority 9) provides a structured reasoning scratchpad. DyTopo swarm (priority 10) launches multi-agent collaboration. The prompt also includes a "Tool selection for external information" subsection categorizing query types (real-time data → Tavily, known URL → Fetch, library docs → Context7, unknown topic → Tavily) and a "Fetch truncation handling" subsection with a 3-truncation circuit breaker.

<!-- Related: tool priority, first-match-wins, priority ladder, rag_search, Memory, Context7, Fetch, Tavily, Desktop Commander, Filesystem, Playwright, DyTopo -->

## When and how does the agent use rag_search for document retrieval?

<!-- Verified: 2026-02-13 -->

LM Studio has no automatic RAG injection — the agent must explicitly call `rag_search` to retrieve information from the `lmstudio_docs` collection on Qdrant port 6334. The system prompt provides domain triggers telling the agent when to search: questions about the stack's configuration, ports, architecture, or VRAM budget; questions about tool usage, operational procedures, or MCP server behavior; questions about previous decisions, conventions, or setup history; uncertainty about how a component works or was deployed; and questions about embedding models, chunking strategy, or collection schemas.

The query technique section instructs the agent to write natural language questions (full sentences retrieve better than keyword fragments in hybrid search), use the `source` filter when the relevant document is known, include both the technical term and a plain-language description for best hybrid matching, and keep limit to 3-5 for targeted queries or 8-10 for broad surveys. The query reformulation section provides guidance when first results are thin: change the question angle, broaden by dropping the source filter, narrow by adding a source filter, or try different vocabulary. The fallback chain progresses from local curated docs outward: `rag_search` → Memory `search_nodes` → Context7 → Tavily → Fetch.

<!-- Related: rag_search, RAG strategy, domain triggers, query technique, source filter, query reformulation, hybrid search, fallback chain, natural language queries -->

## How does the agent launch and manage DyTopo multi-agent swarms?

<!-- Verified: 2026-02-13 -->

The system prompt covers DyTopo (Dynamic Topology) multi-agent swarms through three tools provided by the qdrant-rag server. `swarm_start(task, domain, tau, k_in, max_rounds)` launches a swarm as a background task returning a task_id immediately. `swarm_status(task_id)` checks progress — the agent polls every 15-30 seconds while a swarm runs. `swarm_result(task_id, include_topology)` retrieves the completed output.

Three domains are documented. The code domain (Manager, Developer, Researcher, Tester, Designer) handles code generation, debugging, and architecture. The math domain (Manager, ProblemParser, Solver, Verifier) handles proofs, calculations, and multi-step reasoning. The general domain (Manager, Analyst, Critic, Synthesizer) handles open-ended analysis and multi-perspective reasoning.

The parameter tuning section explains tau (routing threshold τ, default 0.3): lower values (0.1-0.2) create denser inter-agent communication with richer collaboration but higher token cost, higher values (0.4-0.6) create sparser routing with faster convergence and more independent agents. k_in (default 3) caps inbound connections per agent, and max_rounds (default 5, maximum 10) limits iterations. The prompt instructs the agent to handle simple lookups and direct tool calls locally, reserving swarms for tasks where collaborative complexity produces measurably better results.

<!-- Related: DyTopo, swarm_start, swarm_status, swarm_result, domains, tau, k_in, max_rounds, parameter tuning, polling, code, math, general -->

## How does the agent prevent infinite loops and recover from errors?

<!-- Verified: 2026-02-13 -->

The system prompt includes explicit loop prevention and error recovery rules because LM Studio provides no automatic iteration limit, JSON repair, context compression, loop deduplication, or error recovery framework — the agent must self-regulate entirely through prompt-based discipline.

### Loop prevention
The **two-strike rule**: same tool + same arguments + same result twice triggers an immediate change in approach — rephrase the query, switch tools, or report findings. The **progress gate**: every 3 tool calls, confirm new information was gained; if stalled, reassess strategy. The **tool call budget**: simple tasks use 3-5 calls, medium tasks 8-12, complex tasks 15-20. When approaching a budget limit, synthesize findings and deliver a partial answer with a clear statement of what remains unresolved.

### Error classification and recovery
The prompt provides a classify-then-act table: tool timeouts get one retry then switch to an alternative; empty results trigger query rephrasing and scope broadening; missing files trigger Filesystem search then Desktop Commander; permission denied triggers user notification; Qdrant unreachable triggers port status check and docker ps suggestion; parse failures trigger format simplification.

### Fallback chains
Web content: Tavily → Fetch → Playwright. Local knowledge: Memory → rag_search → Filesystem → Desktop Commander. Library docs: Context7 → Tavily → Fetch. The error recovery table also includes a "Fetch truncated" entry: extract the answer if present, otherwise continue; after 3 truncations, switch to Tavily.

<!-- Related: loop discipline, two-strike rule, progress gate, tool call budget, error recovery, fallback chains, self-regulation, classify-then-act, LM Studio limitations -->

## How does the shared Memory knowledge graph protocol work?

<!-- Verified: 2026-02-13 -->

The Memory section of the system prompt is byte-for-byte identical between the LM Studio and AnythingLLM system prompts, ensuring both agents follow the same graph protocol. The shared text specifies that both agents read and write the graph and must maintain consistency.

**What to write:** stable facts, port mappings, collection names, user preferences, project decisions, architecture choices, resolved errors, useful URLs.

**What to skip:** transient context, speculation, secrets, entire file contents (store file paths instead).

**Search-before-create discipline:** before creating entities, always call `search_nodes` first. If the entity exists, use `add_observations` to append new facts rather than creating duplicates. This prevents entity fragmentation that degrades retrieval quality for both agents.

**Naming conventions:** PascalCase entity names (QdrantMCP, BGEm3Config). snake_case entity types (service_config, architecture_decision). snake_case relation names (serves, depends_on, replaced_by). This convention is enforced identically by both agents to keep the shared graph searchable and consistent.

<!-- Related: Memory protocol, shared knowledge graph, search-before-create, PascalCase, snake_case, naming conventions, entity management, graph consistency -->

## How does the agent verify facts, start sessions, and manage output?

<!-- Verified: 2026-02-13 -->

### Verification
The agent confirms facts with tools before stating them: file existence via `search_files`, system state via Desktop Commander, file contents via `read_file`, container health via `docker ps`, and Qdrant connectivity via `rag_status()`. All directives use positive framing ("Confirm X with Y tool") rather than negation.

### Session start
On cold start, the agent orients before acting: (1) `rag_status()` to confirm document freshness and collection health, (2) `search_nodes` in Memory for recent project context and decisions, (3) Desktop Commander to confirm service availability if the task requires live system interaction. Issues discovered during orientation are reported before proceeding.

### Context management
Each tool result is summarized into 1-2 sentences before the next call. Large outputs (logs, file contents, search results) are distilled to relevant facts rather than relying on raw output persisting in context. This preserves the context budget against the RollingWindow manager's message dropping.

### Output and citation
The agent leads with the answer, followed by evidence with tool and source attribution. RAG results cite the source document ("per rag_search on 01-architecture-reference"). Memory results cite the entity. Shell checks cite the command. When tools return URLs (Tavily, Fetch, Context7, Playwright), the agent threads them into the response as inline markdown links — `[display text](url)` — rather than appending separate "Source:" or "Verified via:" blocks. One inline link per source is sufficient. When a tool returns no URL (Memory, Desktop Commander, Filesystem), cite the tool name and key detail. When a fallback chain is exhausted, the agent states which sources were checked and came up empty.

<!-- Related: verification, session start, cold start, context management, output format, citation, source attribution, RollingWindow, positive framing, orientation -->

## What rev4 and rev5 behavioral constraints were added to the system prompt?

<!-- Verified: 2026-02-14 -->

Rev4 introduced three hard behavioral fixes. **Trust hierarchy covers dates** — tool-returned dates are current reality; present without disclaimers. **Response proportionality is a hard constraint** — volunteering unrequested depth is a quality failure. **Follow-up offers are prohibited** — end with the answer. **Citation format:** `[text](url)` markdown links only.

<<<<<<< HEAD
Rev5 added tool-call discipline to prevent two production failures. **Tool-call-first for time-sensitive queries** — call Tavily BEFORE generating any answer text. Generating from training knowledge first creates an anchoring problem where the stale value persists even after the tool returns different data. The correct flow: call tool → read result → build answer from result. **Citation integrity** — only cite a tool when it was actually called in the current turn and the cited data came from that response. Attributing training-knowledge data to Tavily is a hallucinated citation that falsely claims tool-verified authority. The worst failure mode is fabricating a specific source name (e.g., inventing a financial data provider) and presenting training-era data as if a tool returned it — this combines a hallucinated price, a hallucinated source, and a hallucinated citation in a single response. Every value, source name, and URL in the answer must come from the tool result, not from training knowledge or prior prompt examples.
=======
Rev5 added tool-call discipline to prevent two production failures. **Tool-call-first for time-sensitive queries** — call Tavily BEFORE generating any answer text. Generating from training knowledge first creates an anchoring problem where the stale value persists even after the tool returns different data. The correct flow: call tool → read result → build answer from result. **Citation integrity** — only cite a tool when it was actually called in the current turn and the cited data came from that response. Attributing training-knowledge data to Tavily is a hallucinated citation that falsely claims tool-verified authority.
>>>>>>> 9942e327ce1dc149abe142416c07aadc36c3deec

<!-- Related: rev4, rev5, hard constraint, quality failure, trust dates, no follow-up offers, proportionality, citation format, tool-call-first, anchoring, hallucinated citation, citation integrity -->

## What do typical agent workflows look like?

<!-- Verified: 2026-02-13 -->

The system prompt includes workflow examples in compact arrow notation, each demonstrating Act-Observe-Adapt with source citation. The patterns include: **config lookup** (single `rag_search` with citation), **live data request** (`Tavily("current silver price")` → one-call answer with source — added in rev2 to demonstrate Tavily-first for real-time data), **multi-tool chain** (`rag_status` → `rag_reindex`), **DyTopo swarm** (launch → poll → retrieve → present by agent role), **error recovery** (Context7 empty → Tavily fallback), **query reformulation** (broad → narrow with source filter), **Memory integration** (search-before-create), and **file editing** (read → modify → verify).

<<<<<<< HEAD
Each example progresses from trigger condition through tool calls to synthesized answer with source citation. The live data example demonstrates Tavily-first with concise linked output — the response format is `"[Asset] is at [tool-returned price] ([Source](tool-returned-url))."` — a 1-sentence answer matching the question's complexity, built entirely from the tool result. Every bracketed placeholder must be filled exclusively from the Tavily response — the price, the source name, and the URL all come from the tool, never from training knowledge or from examples in this prompt. Rev3 additions include inline markdown link citation for all URL-returning tools and output proportionality guidance scaling response depth to question complexity. Rev4 additions: trust hierarchy covers dates, proportionality upgraded to hard constraint, follow-up offers prohibited, citation format standardized to `[text](url)`. Rev5 additions: tool-call-first discipline for time-sensitive queries (call Tavily before generating answer text to prevent anchoring), citation integrity (only cite tools actually called in the current turn — attributing training data to Tavily is a hallucinated citation). The prompt closes with its bookend: "Use your tools. Present results, not plans. Cite your sources."
=======
Each example progresses from trigger condition through tool calls to synthesized answer with source citation. The live data example demonstrates Tavily-first with concise linked output — `"Silver is trading at $32.45/oz ([Kitco](url))."` — a 1-sentence answer matching the question's complexity. Rev3 additions include inline markdown link citation for all URL-returning tools and output proportionality guidance scaling response depth to question complexity. Rev4 additions: trust hierarchy covers dates, proportionality upgraded to hard constraint, follow-up offers prohibited, citation format standardized to `[text](url)`. Rev5 additions: tool-call-first discipline for time-sensitive queries (call Tavily before generating answer text to prevent anchoring), citation integrity (only cite tools actually called in the current turn — attributing training data to Tavily is a hallucinated citation). The prompt closes with its bookend: "Use your tools. Present results, not plans. Cite your sources."
>>>>>>> 9942e327ce1dc149abe142416c07aadc36c3deec

<!-- Related: workflow examples, config lookup, multi-tool chain, DyTopo swarm, error recovery, query reformulation, Memory integration, web research, file editing, Act-Observe-Adapt, source citation, bookend, rev4, rev5, tool-call-first, anchoring, hallucinated citation -->
