# Memory and Context7 Reference

## How does the Memory knowledge graph work?

<!-- Verified: 2026-02-13 -->

The Memory knowledge graph is an entity-relation store provided through the Docker MCP Gateway as a containerized MCP server. It persists structured facts as named entities connected by typed relations, surviving across sessions and agent restarts. The graph is shared between the LM Studio agent and the AnythingLLM workspace agent — both read and write the same underlying store through the same Docker MCP Memory container. Facts stored by one agent are immediately available to the other. Memory supports `create_entities` for adding nodes with a name, type, and initial observations; `create_relations` for connecting entities with typed edges; `add_observations` for appending new facts to existing entities; `search_nodes` for finding entities by name or observation content; `read_graph` for retrieving the full graph or a subgraph; and `delete_entities` / `delete_relations` for removal. The graph is suited for structured, stable facts that agents need to recall across sessions: port mappings, architecture decisions, user preferences, error solutions, and configuration values.

<!-- Related: Memory, knowledge graph, entities, relations, shared graph, Docker MCP Gateway, persistent facts, search_nodes, create_entities -->

## What naming conventions does the Memory graph use?

<!-- Verified: 2026-02-13 -->

Memory entities use PascalCase names that describe what the entity represents. Examples include QdrantMCP for the MCP qdrant-rag server configuration, BGEm3Config for the BGE-M3 embedding model settings, LmStudioApi for the LM Studio inference endpoint, and DyTopoSwarm for the multi-agent framework configuration. Entity types use snake_case to categorize what kind of thing the entity is: `service_config` for service configurations, `architecture_decision` for design choices, `error_resolution` for discovered fixes, `user_preference` for workflow habits. Relation names use snake_case to describe the connection between entities: `serves` (QdrantMCP serves LmStudioApi), `depends_on` (DyTopoSwarm depends_on LmStudioApi), `replaced_by` (for tracking superseded decisions), `configured_with` (BGEm3Config configured_with specific parameters). This naming convention is enforced identically by both agents. Entity names should be specific enough to distinguish similar entities (QdrantMCP vs QdrantAlm) and descriptive enough that a `search_nodes` query with natural language would find them.

<!-- Related: PascalCase, snake_case, naming conventions, entity names, relation names, entity types, Memory conventions -->

## What should be stored in Memory and what should be skipped?

<!-- Verified: 2026-02-13 -->

Store stable, reusable facts in Memory: port mappings and service endpoints (which service runs on which port), Qdrant collection names and their configurations, user preferences and workflow habits discovered during sessions, project-level architecture decisions and their rationale, error solutions discovered through debugging (so the same error is resolved faster next session), configuration choices and the trade-offs considered, and useful URLs or documentation links relevant to the stack. Skip information that changes frequently or has limited long-term value: transient conversation context that will be irrelevant next session, speculative ideas that have not been confirmed or implemented, secrets and credentials (these must be kept out of the graph entirely), and entire file contents (store the file path instead since files can be read fresh when needed). The goal is a graph of durable reference facts rather than a cache of session state. When in doubt about whether to store something, consider whether it would save meaningful time if recalled in a future session.

<!-- Related: what to store, what to skip, Memory guidelines, stable facts, transient context, secrets, file paths, durable facts -->

## How should Memory be used at session start for cold start orientation?

<!-- Verified: 2026-02-13 -->

At the beginning of a new session, follow a cold start protocol to restore context efficiently. First, call `search_nodes` with broad terms related to the current project context — this recalls known facts about the stack configuration, prior decisions, unresolved issues, and user preferences without requiring the user to re-explain the setup. Second, if the session involves Docker services or live system interaction, run `docker ps` via Desktop Commander to verify which containers are running — this catches cases where Qdrant or other services were restarted or stopped between sessions. Third, call `rag_status()` to verify that the RAG index on Qdrant port 6334 is healthy and up to date — this confirms document freshness before any `rag_search` queries. This three-step protocol — recall from Memory, verify running services, confirm data stores — ensures the agent operates with accurate, current information from the first interaction. Report any issues discovered during orientation (stale index, missing containers, graph inconsistencies) before proceeding with the user's request.

<!-- Related: cold start, session start, orientation, search_nodes, docker ps, rag_status, Memory recall, service verification -->

## How does Context7 work for external library documentation?

<!-- Verified: 2026-02-13 -->

Context7 provides access to current API documentation for external libraries and frameworks through the Docker MCP Gateway. It operates as a two-step process. First, call `resolve-library-id` with the library name (for example, "qdrant-client", "FlagEmbedding", "sentence-transformers", "FastMCP", or "LM Studio") to get the Context7 identifier for that library. Second, call `get-library-docs` with the resolved identifier and an optional topic string to retrieve the relevant documentation sections. Context7 sources documentation that may be newer than the model's training data, making it the preferred tool for version-sensitive questions about library APIs, function signatures, configuration options, method parameters, and breaking changes between versions. For example, if the user asks about a specific qdrant-client method or a FlagEmbedding parameter, Context7 provides the current documentation rather than relying on potentially outdated training knowledge. When Context7 returns documentation with a source URL, cite it as an inline markdown link: `per [Context7 FlagEmbedding docs](url), encode() accepts...`. When no URL is returned, cite as `per Context7 docs for [library]`.

<!-- Related: Context7, resolve-library-id, get-library-docs, external documentation, library API, version-sensitive, current docs -->

## When should Context7 be used versus rag_search versus Memory?

<!-- Verified: 2026-02-13 -->

Each knowledge source serves a distinct purpose with minimal overlap. **Context7** is for external library and API documentation — use it when the question involves a third-party library like qdrant-client, FlagEmbedding, sentence-transformers, FastMCP, LM Studio API, or AnythingLLM API. Context7 provides current, version-accurate documentation that may differ from training data. **rag_search** queries the internal infrastructure documentation about this specific stack — architecture decisions, configuration details, port assignments, chunking strategies, integration patterns, and operational procedures stored in the `lmstudio_docs` collection on Qdrant port 6334. Use `rag_search` when the question is about how this stack is configured, why a design choice was made, or how components interact. **Memory** stores structured facts and preferences discovered during operation — port mappings, entity relationships, error solutions, user preferences, and architecture decisions. Use Memory when recalling a specific stable fact or checking a known configuration value. The general priority: Memory first for known facts (fastest, most specific), then `rag_search` for internal docs (richer context), then Context7 for external docs. For external API questions, skip directly to Context7.

<!-- Related: Context7, rag_search, Memory, knowledge source routing, tool selection, internal vs external docs, priority order -->

## What are the error recovery fallback chains?

<!-- Verified: 2026-02-13 -->

When a tool call fails, classify the error type and follow the corresponding fallback chain rather than immediately reporting failure. For **web content errors** (URL fetch failures, timeouts, 404s): call Tavily first for web search (returns structured results, handles live data), then Fetch for direct URL retrieval if you have a known address, then Playwright if the content requires JavaScript rendering or authentication. For **local knowledge errors** (file missing, service unreachable, permission denied): check Memory for known paths or configurations, then use `rag_search` for relevant internal documentation, then try Filesystem tools for direct file access, then Desktop Commander for shell-level investigation. For **library documentation errors** (Context7 resolve fails, docs unavailable): try Context7 again with an alternative library name, then Tavily for web search, then Fetch for a direct documentation URL. For **connection refused errors** specific to Qdrant or Docker: use Desktop Commander to run `docker ps` to verify the container is running, report the specific port number and service name. Escalate to the user after two different fix attempts have failed — change the approach rather than repeating the same failing strategy.

<!-- Related: error recovery, fallback chains, Fetch, Tavily, Playwright, Memory, rag_search, Filesystem, Desktop Commander, connection refused -->

## How does loop prevention work across tool calls?

<!-- Verified: 2026-02-13 -->

Loop prevention uses three mechanisms to stop unproductive repetition. The **two-strike rule** states that if the same tool is called with the same arguments twice and produces the same result both times, the agent must change its approach immediately — either use a different tool, modify the arguments substantially, or report what was found and what remains unresolved. The **progress gate** triggers every 3 tool calls: the agent evaluates whether meaningful new information was gained in the last 3 calls. If the last 3 calls produced duplicates, empty results, or no progress toward the goal, the agent reassesses strategy before continuing rather than grinding on the same approach. **Budget awareness** sets soft limits on total tool calls per task complexity level: simple tasks (lookups, single-file reads) should complete in 3-5 calls, medium tasks (multi-file investigation, debugging) in 8-12 calls, and complex tasks (architecture analysis, multi-system debugging) in 15-20 calls. Exceeding the budget is a signal to synthesize findings so far, deliver a partial answer, and clearly state what remains unresolved rather than consuming additional calls without progress.

<!-- Related: loop prevention, two-strike rule, progress gate, budget awareness, tool call limits, deduplication, strategy change -->
