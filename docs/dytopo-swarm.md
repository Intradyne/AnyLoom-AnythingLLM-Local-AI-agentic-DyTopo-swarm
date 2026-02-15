# DyTopo Swarm

DyTopo (arXiv 2602.06039) dynamically constructs agent communication topology each round using semantic similarity between agent-generated descriptors.

## Package Architecture

DyTopo is a standalone Python package at `src/dytopo/` with 8 modules. The MCP server (`src/qdrant_mcp_server.py`) exposes 3 thin tools that delegate to it.

| Module | Purpose |
|--------|---------|
| `models.py` | Pydantic v2 data models: `AgentRole`, `SwarmDomain`, `AgentDescriptor`, `AgentState`, `AgentMetrics`, `SwarmMetrics`, `RoundRecord`, `ManagerDecision`, `SwarmStatus`, `SwarmTask` |
| `config.py` | YAML configuration loader — merges `dytopo_config.yaml` over built-in `_DEFAULTS` |
| `agents.py` | System prompts keyed by `(SwarmDomain, AgentRole)`, JSON schemas (`DESCRIPTOR_SCHEMA`, `AGENT_OUTPUT_SCHEMA`, `MANAGER_OUTPUT_SCHEMA`), prompt templates, `build_agent_roster()`, `get_system_prompt()`, `get_role_name()`, `get_worker_names()` |
| `router.py` | Lazy singleton MiniLM-L6-v2, `embed_descriptors()`, `compute_similarity_matrix()`, `apply_threshold()`, `enforce_max_indegree()`, `build_routing_result()`, `log_routing_round()` |
| `graph.py` | `build_execution_graph()` (NetworkX DiGraph), `break_cycles()` (greedy lowest-weight removal), `get_execution_order()` (Kahn's with alphabetical tiebreak), `get_incoming_agents()` |
| `orchestrator.py` | Lazy singleton `AsyncOpenAI` client, `_llm_call()` with tenacity retry (3 attempts, exponential backoff), `_call_manager()`, `_call_worker()`, `run_swarm()` main loop |
| `governance.py` | `execute_agent_safe()`, `detect_convergence()`, `detect_stalling()`, `recommend_redelegation()`, `update_agent_metrics()` |
| `audit.py` | `SwarmAuditLog` class — JSONL event logging to `~/dytopo-logs/{task_id}/audit.jsonl` |

### Configuration

`dytopo_config.yaml` (project root) overrides defaults from `config.py`:

```yaml
llm:
  base_url: "http://localhost:1234/v1"
  model: "qwen3-30b-a3b-instruct-2507"
  # temperature_work: 0.3
  # temperature_descriptor: 0.1
  # temperature_manager: 0.1
routing:
  tau: 0.3
  K_in: 3
orchestration:
  T_max: 5
logging:
  log_dir: "~/dytopo-logs"
```

## Round Lifecycle

1. **Manager** sets round goal (or terminates)
2. **Round 1 — broadcast:** all agents see all outputs (no routing yet)
3. **Rounds 2+ — three-phase split:**
   - Phase A: each agent generates key/query descriptors only (fast, `/no_think`, temp 0.1, 256 tokens)
   - Phase B: MiniLM embeds descriptors -> cosine similarity matrix -> threshold tau -> directed graph -> cycle breaking -> topological sort
   - Phase C: agents execute in topological order with routed messages injected (temp 0.3, 4096 tokens)
4. **Governance checks:** convergence detection, stalling detection, re-delegation recommendations
5. **Audit logging:** all events logged to JSONL for observability

## Three-Phase Architecture

The three-phase split is the key correctness fix: descriptors are generated *before* routing, and work is generated *after* routing with incoming messages injected. Without this, routing would be decorative.

## Agent Domains

| Domain | Manager + Workers | Use case |
|---|---|---|
| `code` | Manager, Developer, Researcher, Tester, Designer | Code generation, debugging, algorithm design |
| `math` | Manager, ProblemParser, Solver, Verifier | Mathematical proofs, calculations |
| `general` | Manager, Analyst, Critic, Synthesizer | Open-ended analysis, multi-perspective reasoning |

## Temperature Differentiation

| Call type | Temperature | Rationale |
|---|---|---|
| Descriptor generation | 0.1 | Near-deterministic structured output for routing accuracy |
| Manager decisions | 0.1 | Consistent goal-setting and termination logic |
| Agent work output | 0.3 | Matches LM Studio default — enough diversity for reasoning |

## MCP Tool Interface

Three tools in `qdrant_mcp_server.py` delegate to the package:

- **`swarm_start(task, domain, tau, k_in, max_rounds)`** — launches background task via `asyncio.create_task()`, returns `task_id` immediately
- **`swarm_status(task_id)`** — reports round progress, elapsed time, LLM calls
- **`swarm_result(task_id, include_topology)`** — returns final answer, metrics, optional per-round topology log

The server stores up to 20 concurrent tasks in `_swarm_tasks` dict and evicts oldest completed tasks when the limit is reached.

## Governance & Robustness

### Failure Recovery
- **Tenacity retry**: `_llm_call()` retries 3 times with exponential backoff on `httpx.TimeoutException` and `openai.APITimeoutError`
- **Worker failure stubs**: `_call_worker()` catches all exceptions and returns an `AgentDescriptor` stub with the error message
- **JSON repair**: `_extract_json()` pipeline: strip `<think>` tags -> direct parse -> markdown fences -> brace-depth extraction -> `json-repair` fallback
- **Never crashes**: Entire `run_swarm()` wrapped in try/except/finally

### Convergence Detection
- **`detect_convergence`**: Compares recent rounds using `difflib.SequenceMatcher`
- **Early termination**: Stops when outputs stabilize (default: 90% similarity over 3 rounds, configurable via `convergence_threshold`)
- **Token savings**: Prevents wasted rounds when solution has converged

### Stalling Detection
- **`detect_stalling`**: Monitors individual agents for repeated outputs
- **Per-agent tracking**: Identifies stuck agents without affecting others
- **Re-delegation triggers**: High stalling similarity triggers `recommend_redelegation()` with actionable recommendations

### Performance Metrics (Pydantic v2)
- **`AgentMetrics`**: `successful_rounds`, `failed_rounds`, `total_latency_ms`, `total_tokens_in/out`, `times_cited`, `times_isolated`
- **Computed properties**: `avg_latency_ms`, `success_rate`, `avg_tokens_per_round`
- **`SwarmMetrics`**: `total_rounds`, `total_llm_calls`, `total_tokens`, `total_wall_time_ms`, `routing_density_per_round`, `convergence_detected_at`, `agent_failures`, `redelegations`, `per_agent` dict

### Audit Logging
- **JSONL format**: One event per line in `~/dytopo-logs/{task_id}/audit.jsonl`
- **Event types**: `swarm_started`, `round_started`, `agent_executed`, `agent_failed`, `convergence_detected`, `redelegation`, `swarm_completed`
- **Routing logs**: Per-round similarity matrices saved as JSON in `~/dytopo-logs/{task_id}/round_NN_routing.json`
- **Analysis-ready**: Easy to parse with `jq`, `awk`, or Python for post-hoc analysis

## Error Isolation

Swarm orchestration runs via `asyncio.create_task()`. Crashes return error strings to the caller — they never propagate to the RAG event loop or affect `rag_search`. The BGE-M3 model, Qdrant client, and MiniLM model are independent singletons that DyTopo code accesses but never mutates. The entire orchestrator is wrapped in try/except/finally to ensure the RAG server never crashes.

## Dependencies

```bash
pip install -r requirements-dytopo.txt
# or individually:
pip install openai>=1.40 tenacity>=9.0 json-repair>=0.39
pip install pyyaml>=6.0 networkx>=3.0 sentence-transformers>=3.0 pydantic>=2.0
```

`sentence-transformers` shares torch with FlagEmbedding — no duplicate install. MiniLM-L6-v2 weights (~80 MB) auto-download from HuggingFace on first swarm.
