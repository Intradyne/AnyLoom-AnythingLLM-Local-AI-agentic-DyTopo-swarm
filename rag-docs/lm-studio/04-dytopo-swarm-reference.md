# DyTopo Multi-Agent Swarm Reference

## What is DyTopo and how does dynamic topology routing work?

<!-- Verified: 2026-02-15 -->

DyTopo (Dynamic Topology) is a multi-agent collaboration framework based on arXiv 2602.06039, implemented as a standalone Python package at `src/dytopo/` with 8 modules. It creates a team of specialized AI agents and routes messages between them using semantically-matched descriptors. The communication graph is rebuilt every round, adapting dynamically based on what agents actually produce and need. Each agent generates short text descriptors: keys describing what it can offer and queries describing what it needs. MiniLM-L6-v2 embeds these descriptors into 384-dimensional vectors on CPU, then a cosine similarity matrix determines which agents should communicate. Edges where similarity exceeds the routing threshold tau (default 0.3) form a directed graph, which is cycle-broken by greedy removal of the lowest-weight edge per cycle and topologically sorted using Kahn's algorithm with alphabetical tiebreaking to establish a deterministic execution order. Round 1 is always a broadcast where all agents see all outputs; starting from round 2, the three-phase descriptor routing takes effect and agents receive only messages from their connected neighbors.

<!-- Related: DyTopo, dynamic topology, multi-agent, routing, arXiv 2602.06039, MiniLM, descriptors, cosine similarity, topological sort -->

## What is the DyTopo package architecture?

<!-- Verified: 2026-02-15 -->

DyTopo is structured as a dedicated Python package at `src/dytopo/` with 8 modules. The `models.py` module defines Pydantic v2 data models including AgentRole enum (11 roles across 3 domains), SwarmDomain enum (code/math/general), AgentDescriptor (key/query/work fields), AgentState (per-agent state with history and metrics), AgentMetrics (per-agent performance tracking with computed properties for avg_latency_ms, success_rate, avg_tokens_per_round), SwarmMetrics (aggregate metrics for the swarm run), RoundRecord, ManagerDecision, SwarmStatus enum, and SwarmTask (the top-level orchestration object with auto-generated task_id). The `config.py` module loads YAML configuration from `dytopo_config.yaml` at the project root, merging user overrides over built-in defaults for llm, routing, orchestration, and logging sections. The `agents.py` module contains system prompts keyed by (SwarmDomain, AgentRole) tuples, three JSON schemas (DESCRIPTOR_SCHEMA, AGENT_OUTPUT_SCHEMA, MANAGER_OUTPUT_SCHEMA), prompt templates (DESCRIPTOR_INJECTION, DESCRIPTOR_ONLY_INSTRUCTIONS, WORK_INSTRUCTIONS, INCOMING_MSG_TEMPLATE, PROMPT_INJECTION_GUARD), and functions build_agent_roster(), get_system_prompt(), get_role_name(), get_worker_names(). The `router.py` module handles MiniLM-L6-v2 embedding and similarity routing. The `graph.py` module handles NetworkX DAG construction, cycle breaking, and topological sort. The `orchestrator.py` module contains the main swarm loop with a lazy singleton AsyncOpenAI client and tenacity retry logic. The `governance.py` module provides convergence detection, stalling detection, and re-delegation recommendations. The `audit.py` module writes JSONL audit logs. The MCP server (`src/qdrant_mcp_server.py`) exposes 3 thin tools (swarm_start, swarm_status, swarm_result) that delegate to this package.

<!-- Related: package architecture, src/dytopo, models.py, config.py, agents.py, router.py, graph.py, orchestrator.py, governance.py, audit.py, Pydantic v2, YAML config, module structure -->

## What agent teams are available in each DyTopo domain?

<!-- Verified: 2026-02-15 -->

DyTopo organizes agents into three predefined domains, each tailored to a category of problem. The **code** domain contains a Manager plus four workers: Developer, Researcher, Tester, and Designer — suited for code generation, debugging, algorithm design, software architecture, and code review tasks. The **math** domain contains a Manager plus three workers: ProblemParser, Solver, and Verifier — designed for proofs, numerical calculations, optimization problems, and multi-step mathematical reasoning where cross-verification catches errors. The **general** domain contains a Manager plus three workers: Analyst, Critic, and Synthesizer — intended for open-ended analysis, multi-perspective reasoning, evaluating trade-offs, and tasks requiring balanced critique and synthesis. Every domain includes a Manager agent responsible for defining round goals, deciding when to terminate (setting `terminate=true` when the team consensus is reached or after 3+ rounds without progress), and extracting the final solution from the collaborative output. Agent rosters are built by `build_agent_roster(domain)` in `agents.py`, and system prompts are stored in the `SYSTEM_PROMPTS` dict keyed by `(SwarmDomain, AgentRole)` tuples.

<!-- Related: domains, code, math, general, Manager, Developer, Researcher, Tester, Designer, ProblemParser, Solver, Verifier, Analyst, Critic, Synthesizer, build_agent_roster, SYSTEM_PROMPTS -->

## How does the three-phase round structure work in DyTopo?

<!-- Verified: 2026-02-15 -->

Starting from round 2, each DyTopo round executes three distinct phases. In **Phase A (descriptor generation)**, each agent generates short key and query descriptors using Qwen3-30B-A3B at temperature 0.1 with the `/no_think` flag and a 256-token output limit — this keeps descriptor generation fast, cheap, and deterministic. The `_call_worker()` function in `orchestrator.py` handles this with `descriptor_only=True`, using `DESCRIPTOR_SCHEMA` from `agents.py`. In **Phase B (topology routing)**, the `build_routing_result()` function in `router.py` embeds all descriptors via MiniLM-L6-v2, computes a cosine similarity matrix S where S[i][j] = cosine(query_i, key_j), applies the adjacency threshold `A = (S >= tau)`, enforces k_in by keeping only the top-K highest-similarity inbound edges per agent, then `graph.py` breaks cycles by greedy removal of the lowest-weight edge and performs topological sort via Kahn's algorithm with alphabetical tiebreaking. In **Phase C (work execution)**, agents execute their main work in the topological order from Phase B, with routed messages from connected agents injected into their context via `_gather_routed()`. Phase C uses temperature 0.3 with a 4096-token output limit and `AGENT_OUTPUT_SCHEMA`. The Manager agent runs at temperature 0.1 with a 2000-token limit using `MANAGER_OUTPUT_SCHEMA`.

<!-- Related: Phase A, Phase B, Phase C, descriptor generation, topology routing, work execution, three-phase, temperature, MiniLM routing, _call_worker, build_routing_result, DESCRIPTOR_SCHEMA, AGENT_OUTPUT_SCHEMA -->

## What parameters control DyTopo swarm behavior and how should they be tuned?

<!-- Verified: 2026-02-15 -->

Three main parameters control how a DyTopo swarm operates, configurable via `dytopo_config.yaml` or as MCP tool arguments. The **tau** parameter (routing threshold, default 0.3) sets the cosine similarity threshold — edges above tau create communication links between agents. A tau of 0.2 creates dense connections where most agents communicate with most others, producing richer collaboration at higher token cost. A tau of 0.3 (default) balances breadth of communication with focused relevance. A tau of 0.5 creates sparse connections where agents only link when descriptors are highly similar, producing faster convergence with more independent agents. The **k_in** parameter (default 3, range 1-5) caps the maximum inbound messages any single agent receives per round, preventing information overload on popular agents. The **max_rounds** (T_max, default 5, range 1-10) limits total swarm rounds. A convergence check runs each round (using `detect_convergence()` from `governance.py`): if agent outputs stop changing meaningfully (default 90% similarity over 3 rounds, configurable via `convergence_threshold`), or the Manager sets `terminate=true`, the swarm terminates early. Temperature settings are in `dytopo_config.yaml` under the `llm` section: `temperature_work` (0.3), `temperature_descriptor` (0.1), `temperature_manager` (0.1).

<!-- Related: tau, k_in, max_rounds, T_max, routing threshold, parameter tuning, convergence, termination, DyTopo configuration, dytopo_config.yaml, convergence_threshold -->

## How should a DyTopo swarm be launched and monitored?

<!-- Verified: 2026-02-15 -->

DyTopo is controlled through three MCP tools in `qdrant_mcp_server.py` that delegate to the `src/dytopo/` package. Call `swarm_start(task, domain, tau, k_in, max_rounds)` to launch — it creates a `SwarmTask` Pydantic object, fires off `run_swarm()` via `asyncio.create_task()`, and returns a `task_id` immediately without blocking. Only `task` is required; `domain` defaults to "code", and tau, k_in, max_rounds use defaults of 0.3, 3, and 5 respectively. Poll `swarm_status(task_id)` every 15-30 seconds to check progress — it returns the current round number, LLM call count, elapsed wall-clock time, progress message, and overall status (running, completed, or failed). When status shows completed, call `swarm_result(task_id, include_topology)` to retrieve the final output including SwarmMetrics (total rounds, tokens, wall time, routing density, convergence point, per-agent metrics). Set `include_topology=true` for debugging — it adds a per-round log showing edges, similarity scores, and execution order. The server stores up to 20 concurrent tasks and evicts the oldest half of completed tasks when the limit is reached.

<!-- Related: swarm_start, swarm_status, swarm_result, task_id, polling, monitoring, launch workflow, AsyncOpenAI, SwarmTask, asyncio.create_task -->

## When should a DyTopo swarm be used instead of solving directly?

<!-- Verified: 2026-02-15 -->

DyTopo swarms are valuable for complex, multi-perspective problems where specialist viewpoints improve the result through iterative collaboration. Good use cases include code review and debugging (Developer + Tester + Researcher collaborate on finding and fixing issues), multi-step mathematical reasoning (Parser + Solver + Verifier cross-check each other's work), architecture design (multiple specialists contribute domain expertise), and open-ended analysis that benefits from a critic-synthesizer dynamic. DyTopo adds meaningful overhead — multiple rounds of inference at localhost:1234, descriptor generation, and routing computation — so it should be reserved for tasks where that overhead produces better results than a single-pass response. Handle simple lookups, single-tool tasks, factual questions answerable from RAG or Memory, and time-sensitive queries directly. A direct response from the agent is faster and sufficient for straightforward requests. Launch a swarm when the problem genuinely benefits from multiple agents iterating toward a refined, cross-verified answer.

<!-- Related: when to use DyTopo, use cases, overhead, code review, math proofs, architecture, multi-perspective analysis -->

## How is DyTopo isolated from the RAG pipeline?

<!-- Verified: 2026-02-15 -->

DyTopo swarms run via `asyncio.create_task()` in the qdrant-rag server, which isolates them from the main server event loop. The entire `run_swarm()` function in `orchestrator.py` is wrapped in try/except/finally — if a swarm crashes or encounters an error, it returns an error string to the caller and the exception is contained. The `_safe_run()` wrapper in the MCP tool catches any exception from `run_swarm()` and stores it in `_swarm_tasks` as an error status. The three core server singletons — the BGE-M3 FlagEmbedding model for RAG embedding, the Qdrant client connected to port 6334, and the MiniLM-L6-v2 routing model — are independent of the swarm lifecycle. DyTopo accesses MiniLM-L6-v2 in read-only mode via its own lazy singleton in `router.py` and accesses the LM Studio API via a separate lazy singleton AsyncOpenAI client in `orchestrator.py`. A swarm failure cannot corrupt the Qdrant collection, disrupt embedding generation, or block incoming `rag_search` requests.

<!-- Related: isolation, asyncio, crash safety, BGE-M3, MiniLM, Qdrant client, RAG stability, concurrent swarms, _safe_run -->

## What models does DyTopo use and why was MiniLM chosen for routing?

<!-- Verified: 2026-02-15 -->

DyTopo uses two models with distinct roles, both configured in `dytopo_config.yaml`. **Qwen3-30B-A3B-Instruct-2507** (Q6_K) at localhost:1234 handles all agent inference via a lazy singleton AsyncOpenAI client in `orchestrator.py`. Descriptor generation in Phase A runs at temperature 0.1 with `/no_think` and a 256-token limit. Work execution in Phase C runs at temperature 0.3 with a 4096-token limit. Manager decisions use temperature 0.1 with a 2000-token limit. All inference calls include sampling parameters top_p=0.85, top_k=20, and min_p=0.05. The `_llm_call()` function uses tenacity retry with 3 attempts and exponential backoff (1-4s) on timeout exceptions. **MiniLM-L6-v2** (22M parameters, 384-dimensional output, ~80 MB RAM on CPU) handles descriptor routing in Phase B via a lazy singleton in `router.py`. It embeds key and query vectors used to build the communication graph via cosine similarity. MiniLM-L6-v2 was chosen over BGE-M3 for routing because its wider similarity spread provides better threshold discrimination at the tau boundary.

<!-- Related: Qwen3, MiniLM-L6-v2, inference model, routing model, temperature, similarity spread, tau discrimination, model choice, _llm_call, tenacity, AsyncOpenAI -->

## How does DyTopo handle agent failures and ensure robustness?

<!-- Verified: 2026-02-15 -->

DyTopo implements comprehensive failure recovery across multiple layers. The `_llm_call()` function in `orchestrator.py` uses tenacity retry with 3 attempts and exponential backoff (1-4 seconds) on `httpx.TimeoutException` and `openai.APITimeoutError`, plus `asyncio.wait_for()` with configurable timeout. The `_call_worker()` function catches all exceptions and returns a failure stub `AgentDescriptor` containing the error message, allowing other agents to continue working. The `_extract_json()` pipeline handles malformed LLM outputs through 5 stages: strip `<think>` tags, direct JSON parse, markdown code fence extraction, brace-depth extraction, and `json-repair` fallback. The `governance.py` module provides `detect_convergence()` and `recommend_redelegation()` which the orchestrator calls after each round to identify stalling agents and suggest prompt modifications. The entire `run_swarm()` is wrapped in try/except/finally to ensure the audit log is closed and swarm metrics are updated even on failure. All exceptions are logged and stored as error strings in SwarmTask.

<!-- Related: failure recovery, tenacity retry, timeout, json-repair, graceful degradation, error handling, robustness, _call_worker, _extract_json, _llm_call -->

## What is convergence detection and when does it trigger?

<!-- Verified: 2026-02-15 -->

Convergence detection monitors swarm outputs across recent rounds to determine when the solution has stabilized. After each round (starting from round 3), the orchestrator calls `detect_convergence()` from `governance.py`, which compares agent outputs across a sliding window (default: last 3 rounds) using `difflib.SequenceMatcher` to compute text similarity. If the similarity exceeds the threshold (default: 90%, configurable via `convergence_threshold` in `dytopo_config.yaml` under the `orchestration` section), the swarm terminates early with `swarm.termination_reason = "convergence"`. This saves tokens and wall-clock time by preventing wasted rounds when agents have already reached consensus. Convergence is tracked in `SwarmMetrics.convergence_detected_at` (stores the round number) and logged to the audit log via `audit.convergence_detected()`. Convergence detection is separate from the Manager's termination decision — both can trigger early termination independently.

<!-- Related: convergence detection, detect_convergence, difflib, SequenceMatcher, early termination, similarity threshold, token savings, convergence_threshold, orchestration config -->

## What performance metrics does DyTopo track and how are they computed?

<!-- Verified: 2026-02-15 -->

DyTopo tracks metrics at two levels using Pydantic v2 models in `models.py` with computed properties. **Per-agent metrics** (`AgentMetrics`) include: `successful_rounds` and `failed_rounds` (execution outcomes), `total_latency_ms` (cumulative response time), `total_tokens_in` and `total_tokens_out` (LLM usage), `times_cited` (how many agents received this agent's output), `times_isolated` (rounds with zero incoming edges), and `descriptor_quality_scores` (similarity to best match). Three computed properties auto-calculate derived values: `avg_latency_ms = total_latency_ms / (successful + failed)`, `success_rate = successful / (successful + failed)`, and `avg_tokens_per_round = (tokens_in + tokens_out) / successful`. **Swarm-level metrics** (`SwarmMetrics`) include: `total_rounds`, `total_llm_calls`, `total_tokens`, `total_wall_time_ms`, `routing_density_per_round` (edges per round), `convergence_detected_at` (round number or None), `cycle_breaks`, `isolation_fallbacks`, `agent_failures`, `redelegations`, and `per_agent` dict mapping agent IDs to their `AgentMetrics`. All metrics are included in the `swarm_result` MCP tool output.

<!-- Related: metrics, AgentMetrics, SwarmMetrics, computed properties, performance tracking, success_rate, avg_latency, tokens, routing_density, Pydantic v2 -->

## What is audit logging and how can logs be analyzed?

<!-- Verified: 2026-02-15 -->

DyTopo writes comprehensive execution traces to JSONL (JSON Lines) files via the `SwarmAuditLog` class in `audit.py`. Each swarm creates a directory at `~/dytopo-logs/{task_id}/` containing `audit.jsonl`, where each line is a self-contained JSON object representing one event. Event types include: `swarm_started` (task description, max rounds, agent list), `round_started` (round number), `agent_executed` (agent name, output, execution time), `agent_failed` (agent name, error message), `convergence_detected` (round number, reason, confidence score), `redelegation` (from agent, reason, recommendation), and `swarm_completed` (final output, total rounds). Additionally, when `save_similarity_matrices` is enabled in config (default true), per-round routing data is saved to `~/dytopo-logs/{task_id}/round_NN_routing.json` via `log_routing_round()` in `router.py`. Logs can be analyzed with standard Unix tools: `cat audit.jsonl | jq '.event_type' | sort | uniq -c` counts event types, `jq 'select(.event_type == "agent_failed")'` filters failures, and routing JSON files contain full similarity matrices for post-hoc analysis.

<!-- Related: audit logging, SwarmAuditLog, JSONL, event types, observability, jq, analysis, debugging, execution trace, log_routing_round, routing JSON -->

## What is stalling detection and re-delegation?

<!-- Verified: 2026-02-15 -->

Stalling detection monitors individual agents for repeated outputs that indicate the agent is stuck. The `detect_stalling()` function in `governance.py` compares a specific agent's outputs across recent rounds (default: last 3) using `difflib.SequenceMatcher`. If similarity exceeds the threshold (default: 98%), the agent is flagged as stalling. Unlike convergence detection (which looks at all agents), stalling detection focuses on one agent at a time. The orchestrator calls `recommend_redelegation()` after round 2+, which analyzes the round history and agent metrics to generate actionable recommendations. It identifies agents that are stalling (>98% similarity), frequently failing (failure_count >= threshold), or isolated (zero incoming edges across multiple rounds). Each recommendation includes the `agent_id`, a `reason`, a specific `recommendation`, and a `severity` level (high/medium/low). Re-delegation events are logged to the audit log via `audit.redelegation()` and counted in `SwarmMetrics.redelegations`.

<!-- Related: stalling detection, detect_stalling, recommend_redelegation, agent stuck, prompt modification, redelegation, severity, recommendations, governance.py -->
