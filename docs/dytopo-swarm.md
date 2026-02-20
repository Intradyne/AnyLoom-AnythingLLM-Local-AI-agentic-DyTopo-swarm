# DyTopo Swarm

DyTopo (arXiv 2602.06039) dynamically constructs agent communication topology each round using semantic similarity between agent-generated descriptors.

## Package Architecture

DyTopo is a standalone Python package at `src/dytopo/` with 12 core modules and 6 sub-packages. The MCP server (`src/qdrant_mcp_server.py`) exposes 3 swarm tools (`swarm_start`, `swarm_status`, `swarm_result`) plus 5 RAG tools that delegate to it.

### Core Modules

| Module | Purpose |
|--------|---------|
| `models.py` | Pydantic v2 data models: `AgentRole`, `SwarmDomain`, `AgentDescriptor`, `AgentState`, `AgentMetrics`, `SwarmMetrics`, `RoundRecord`, `ManagerDecision`, `SwarmStatus`, `SwarmTask`, `SwarmMemoryRecord`, `HealthStatus`, `StackHealth`, `AegeanVote` |
| `config.py` | YAML configuration loader — merges `dytopo_config.yaml` over built-in `_DEFAULTS`; includes `concurrency` section for backend selection |
| `agents.py` | System prompts keyed by `(SwarmDomain, AgentRole)`, JSON schemas (`DESCRIPTOR_SCHEMA`, `AGENT_OUTPUT_SCHEMA`, `MANAGER_OUTPUT_SCHEMA`), prompt templates, `build_agent_roster()`, `get_system_prompt()`, `get_role_name()`, `get_worker_names()` |
| `router.py` | Lazy singleton MiniLM-L6-v2, `embed_descriptors()`, `compute_similarity_matrix()`, `apply_threshold()`, `enforce_max_indegree()`, `build_routing_result()`, `log_routing_round()`, `prepare_descriptor_for_embedding()` (intent enrichment), `validate_descriptor_separation()` |
| `stigmergic_router.py` | `StigmergicRouter` class wrapping functional routing with trace-aware topology: `build_topology()` (blends similarity matrix with time-decayed historical trace boost), `deposit_trace()` (persists swarm routing patterns to Qdrant `swarm_traces` collection), `get_trace_stats()`, `prune_old_traces()`. `build_trace_edges()` helper converts routing edges to trace format. Uses 384-dim MiniLM-L6-v2 embeddings, configurable via `traces` section in config |
| `graph.py` | `build_execution_graph()` (NetworkX DiGraph), `break_cycles()` (greedy lowest-weight removal), `get_execution_order()` (Kahn's with alphabetical tiebreak), `get_execution_tiers()` (parallel-within-tier ordering via `nx.topological_generations()`), `get_incoming_agents()` |
| `orchestrator.py` | Backend-agnostic LLM client via `_get_llm_client()` (connects to llama.cpp on port 8008), semaphore-based concurrency via `_get_semaphore()` (8 concurrent), `_llm_call()` with tenacity retry (3 attempts, exponential backoff), `_call_manager()`, `_call_worker()`, `run_swarm()` main loop with parallelized phases via `asyncio.gather()`. Integrates checkpoint, policy, verifier, and stalemate modules via guarded imports (`_HAS_CHECKPOINT`, `_HAS_STALEMATE`, `_HAS_VERIFIER`, `_HAS_POLICY` flags) |
| `governance.py` | `execute_agent_safe()`, `detect_convergence()`, `detect_stalling()`, `recommend_redelegation()`, `update_agent_metrics()`, `compute_consensus_matrix()`, `check_aegean_termination()`, `StalemateDetector` (ping-pong, no-progress, regression detection), `StalemateResult`, `get_generalist_fallback_agent()` |
| `checkpoint.py` | `CheckpointManager` — atomic crash-recovery persistence of `SwarmTask` state. Uses `os.replace()` for atomic writes, Pydantic v2 `model_dump(mode="json")` / `model_validate()` round-trip, envelope format with version and timestamps. Methods: `save()`, `load_latest()`, `list_hot_tasks()`, `mark_completed()`, `cleanup()` |
| `policy.py` | `PolicyEnforcer` (PCAS-Lite) — deny-first tool-call policy enforcement from `policy.json`. Evaluates file read/write, shell exec, and network operations. Path traversal prevention via resolved absolute paths. `PolicyDecision` dataclass. Methods: `check_tool_request()`, `enforce()` |
| `verifier.py` | `OutputVerifier` — deterministic output verification (no LLM calls). Dispatches to `_check_python_syntax()` (via `ast.parse()`), `_check_schema()` (JSON required-fields validation), `_execute_code()` (sandboxed subprocess). Fail-open policy: infrastructure errors default to PASS. `VerificationResult` dataclass |
| `audit.py` | `SwarmAuditLog` class — JSONL event logging to `~/dytopo-logs/{task_id}/audit.jsonl` |

### Sub-packages

| Sub-package | Key classes / modules | Purpose |
|-------------|----------------------|---------|
| `observability/` | `TraceContext`, `MetricsCollector`, `PerformanceProfiler`, `BottleneckAnalyzer`, `FailureAnalyzer` | Distributed tracing via `contextvars`, percentile metrics with Prometheus export, performance profiling, failure analysis |
| `safeguards/` | `RateLimiter`, `TokenBudget`, `CircuitBreaker`, `PerformanceSafeguards` | Rate limiting, token budget enforcement, circuit-breaker pattern for LLM calls |
| `messaging/` | `AgentMessage`, `MessageHistory`, `MessageRouter` | Typed message passing between agents with history tracking |
| `routing/async_engine.py` | `AsyncRoutingEngine` | Async wrapper around `router.py` with lock-based embedding serialization |
| `delegation/` | `DelegationManager`, `DelegationRecord` | Delegation with depth, concurrency, and timeout control |
| `memory/` | `SwarmMemoryWriter` | Embeds and persists completed swarm results to Qdrant for semantic retrieval of past solutions |
| `health/` | `HealthChecker`, `preflight_check()` | Parallel health probes for LLM, Qdrant, AnythingLLM, and GPU; pre-run preflight gate |
| `documentation/` | `DocumentationGenerator` | Auto-generated living docs from code and execution data |

### Configuration

`dytopo_config.yaml` (project root) overrides defaults from `config.py`:

```yaml
llm:
  base_url: "http://localhost:8008/v1"  # llama.cpp Docker container
  model: "qwen3-30b-a3b-instruct-2507"
  # temperature_work: 0.3
  # temperature_descriptor: 0.1
  # temperature_manager: 0.1
routing:
  tau: 0.3
  K_in: 3
orchestration:
  T_max: 5
concurrency:
  backend: "llama-cpp"      # llama.cpp (parallel, default)
  max_concurrent: 8         # 8 concurrent requests
  # connect_timeout: 10.0
  # read_timeout: 180.0
logging:
  log_dir: "~/dytopo-logs"
traces:
  enabled: true             # Enable stigmergic trace-aware routing
  qdrant_url: "http://localhost:6333"
  collection: "swarm_traces"
  boost_weight: 0.15        # Alpha blending: S = (1-α)*S + α*B
  half_life_hours: 168.0    # Time decay half-life (1 week)
  top_k: 5                  # Number of historical traces to retrieve
  min_quality: 0.5          # Quality gate for depositing traces
  prune_max_age_hours: 720.0  # Delete traces older than 30 days
health_monitor:
  check_interval_seconds: 30
  max_restart_attempts: 3
  crash_window_minutes: 15
  alert_cooldown_minutes: 30
  log_dir: "~/anyloom-logs"
checkpoint:
  enabled: true               # Enable checkpoint crash recovery
  checkpoint_dir: "~/dytopo-checkpoints"
  save_per_agent: false        # Save per-agent checkpoints (verbose)
verification:
  enabled: false               # Enable deterministic output verification
  max_retries: 1               # Retries before accepting failed verification
  specs:                        # Per-role verification specs
    developer:
      type: "syntax_check"     # ast.parse() Python syntax check
      timeout_seconds: 10
    researcher:
      type: "schema_validation"
      required_fields: ["sources", "summary"]
    solver:
      type: "syntax_check"
      timeout_seconds: 30
```

## Round Lifecycle

1. **Manager** sets round goal (or terminates)
2. **Round 1 — broadcast:** all agents see all outputs (no routing yet); agent calls are parallelized via `asyncio.gather()`
3. **Rounds 2+ — three-phase split:**
   - Phase A: each agent generates key/query descriptors only (fast, `/no_think`, temp 0.1, 256 tokens); descriptor generation is parallelized via `asyncio.gather()`
   - Phase B: MiniLM embeds descriptors (with `prepare_descriptor_for_embedding()` intent enrichment) -> cosine similarity matrix -> optional stigmergic trace boost (blends historical routing patterns via time-decayed role-pair matrix) -> threshold tau -> directed graph -> cycle breaking -> topological tiers via `get_execution_tiers()`. Optional `validate_descriptor_separation()` check for routing quality
   - Phase C: agents execute tier-by-tier with `asyncio.gather()` within each tier; agents in the same tier run in parallel, later tiers wait for earlier tiers to complete; routed messages injected (temp 0.3, 4096 tokens). If verification is enabled, each agent output passes through `OutputVerifier` (syntax check / schema validation / code execution). If policy enforcement is enabled, tool calls are gated by `PolicyEnforcer` before execution. Checkpoints saved after each step when enabled
4. **Governance checks:** convergence detection, stalling detection, re-delegation recommendations, stalemate detection (ping-pong, no-progress, regression patterns)
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
| Agent work output | 0.3 | Good balance — enough diversity for reasoning without sacrificing precision |

## MCP Tool Interface

The `qdrant_mcp_server.py` exposes 8 tools total. Three swarm tools delegate to the DyTopo package:

- **`swarm_start(task, domain, tau, k_in, max_rounds)`** — launches background task via `asyncio.create_task()`, returns `task_id` immediately
- **`swarm_status(task_id)`** — reports round progress, elapsed time, LLM calls
- **`swarm_result(task_id, include_topology)`** — returns final answer, metrics, optional per-round topology log

Five RAG tools provide hybrid search: `rag_search`, `rag_status`, `rag_reindex`, `rag_sources`, `rag_file_info`.

A separate **system-status MCP server** (`src/mcp_servers/system_status_mcp.py`) provides 6 diagnostic tools: `service_health`, `qdrant_collections`, `gpu_status`, `llm_slots`, `docker_status`, `stack_config`.

The swarm server stores up to 20 concurrent tasks in `_swarm_tasks` dict and evicts oldest completed tasks when the limit is reached.

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

### Aegean Consensus Termination
- **`check_aegean_termination`**: Embedding-based consensus check across agent outputs using MiniLM-L6-v2
- **Algorithm**: Embeds all agent outputs, computes pairwise cosine similarity, each agent "votes" to terminate if its avg similarity exceeds `consensus_threshold` (default 0.85). If >= 75% of agents vote, the swarm terminates early
- **Integration**: Runs after convergence detection in rounds >= 2; non-fatal — errors are silently logged

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

### Swarm Memory Persistence
- **`SwarmMemoryWriter`** (`memory/writer.py`): After a successful swarm run, the task description, key findings, and agent outputs are embedded via MiniLM-L6-v2 (384-dim) and stored in Qdrant collection `swarm_memory`
- **Semantic retrieval**: `query_similar(text, limit)` finds past solutions relevant to a new task
- **Graceful degradation**: Qdrant unavailability is caught and logged — memory write failure never crashes a swarm

### Stigmergic Trace Routing
- **`StigmergicRouter`** (`stigmergic_router.py`): Wraps the functional routing pipeline and adds trace-aware topology construction via Qdrant-persisted swarm traces
- **Trace deposit**: After a successful swarm, the routing pattern (active edges, agent roles, convergence data, quality score) is embedded via MiniLM-L6-v2 (384-dim) and stored in Qdrant collection `swarm_traces`
- **Trace retrieval**: On topology construction, similar past traces are retrieved and used to compute a time-decayed boost matrix. The boost blends with the cosine similarity matrix: `S = (1-α)*S + α*B` where `α = boost_weight` (default 0.15)
- **Role-pair mapping**: Historical traces map role-pairs (not agent IDs), so traces from one swarm can boost routing in a differently-composed swarm with matching roles
- **Time decay**: Trace influence decays exponentially: `weight = 2^(-age_hours / half_life_hours)` (default half-life: 168h / 1 week)
- **Quality gating**: Only traces with `final_answer_quality >= min_quality` (default 0.5) are deposited
- **Graceful degradation**: All Qdrant operations are wrapped in try/except — trace failure never affects routing correctness
- **Pruning**: `prune_old_traces()` deletes traces older than `prune_max_age_hours` (default 720h / 30 days)

### Pre-run Health Check
- **`preflight_check()`** (`health/checker.py`): Parallel HTTP probes to LLM (llama.cpp `/v1/models`), Qdrant (`/collections`), AnythingLLM (`/api/v1/auth`), and GPU (`nvidia-smi`)
- **Integration**: Runs at the start of `run_swarm()`. If LLM is unreachable, the swarm aborts with `RuntimeError`. Other component failures are logged as warnings
- **Models**: `HealthStatus` and `StackHealth` Pydantic models track per-component health with latency
- **Health Monitor Sidecar** (`scripts/health_monitor.py`): Standalone Python process that reuses `HealthChecker` for continuous monitoring (every 30s), auto-restarts failed Docker containers, and logs structured JSONL. Separate from the swarm — runs independently alongside the Docker stack

### Stalemate Detection

The `StalemateDetector` class in `governance.py` complements per-agent `detect_stalling()` by analyzing *routing patterns* across rounds rather than individual agent outputs. It detects three patterns:

1. **Ping-pong**: Two agents routing to each other repeatedly without involving others. Suggests injecting a generalist fallback agent via `get_generalist_fallback_agent()`.
2. **No progress**: Convergence score unchanged (delta < 0.01) for multiple rounds. Suggests forced termination.
3. **Regression**: Convergence score actively decreasing, indicating the swarm is diverging. Suggests human-in-the-loop intervention.

Returns a `StalemateResult` dataclass with `is_stalled`, `reason`, `suggested_action`, and `stale_pair`. The orchestrator calls `StalemateDetector.check()` after governance checks when `_HAS_STALEMATE` is True.

### Checkpoint Recovery

The `CheckpointManager` class in `checkpoint.py` persists `SwarmTask` state to disk after each orchestration step so that a crashed or interrupted run can be resumed from the last good checkpoint.

- **Atomic writes**: Each save writes to a temp file, then calls `os.replace()` (atomic on POSIX, near-atomic on Windows same-volume rename).
- **Pydantic v2 round-trip**: `model_dump(mode="json")` for serialization, `model_validate()` for deserialization.
- **Envelope format**: Every checkpoint JSON carries `__checkpoint_version__`, `__step_label__`, and ISO-8601 `__timestamp__` metadata alongside the task payload.
- **Hot task scanning**: `list_hot_tasks()` finds incomplete tasks (no `_completed` marker) for automated resume.
- **Configuration**: Controlled via `checkpoint` section in `dytopo_config.yaml`. Enabled by default; checkpoints stored in `~/dytopo-checkpoints/`.

### Policy Enforcement

The `PolicyEnforcer` class in `policy.py` implements PCAS-Lite (Policy-Controlled Agent Sandbox, Lite edition) for deny-first tool-call policy enforcement.

- **Policy file**: Rules loaded from `policy.json` at the project root. Defines allow/deny patterns for file read/write, shell exec, and network operations.
- **Deny-first evaluation**: Deny rules are checked first; if ANY deny matches, the request is blocked immediately. Allow rules are checked next. Default action is deny if no allow rule matches.
- **Path traversal prevention**: All file paths are resolved to absolute before matching, preventing `../` escape attacks.
- **Integration**: The orchestrator checks `PolicyEnforcer.enforce()` before executing tool calls when `_HAS_POLICY` is True. Returns `None` if allowed, or an error dict if denied.

### Silent Verification

The `OutputVerifier` class in `verifier.py` provides deterministic output verification for agent work products without any LLM calls.

- **Verification strategies**: `syntax_check` (Python syntax via `ast.parse()`), `schema_validation` (JSON required-fields check), `code_execution` (sandboxed subprocess with timeout).
- **Fail-open policy**: Any infrastructure error results in `passed=True` so the swarm pipeline is never blocked by the verifier itself.
- **Code block extraction**: Automatically extracts fenced code blocks from markdown output before verification.
- **Per-role configuration**: Verification specs are configured per agent role in the `verification.specs` section of `dytopo_config.yaml`.
- **Integration**: The orchestrator runs `OutputVerifier.verify()` after each agent execution in Phase C when `_HAS_VERIFIER` is True. Failed verifications are logged but do not crash the pipeline.

## Error Isolation

Swarm orchestration runs via `asyncio.create_task()`. Crashes return error strings to the caller — they never propagate to the RAG event loop or affect `rag_search`. The BGE-M3 model, Qdrant client, and MiniLM model are independent singletons that DyTopo code accesses but never mutates. The entire orchestrator is wrapped in try/except/finally to ensure the RAG server never crashes.

## Observability Layer

The `observability/` sub-package provides production-grade instrumentation for swarm runs:

- **TraceContext**: Distributed tracing via `contextvars` — propagates trace and span IDs through async call chains without explicit parameter threading.
- **MetricsCollector**: Collects latency, token usage, and routing density metrics with percentile breakdowns (p50/p95/p99). Supports Prometheus export format.
- **PerformanceProfiler**: Wall-clock and per-phase profiling for identifying slow rounds and bottleneck agents.
- **BottleneckAnalyzer**: Identifies agents or phases that consistently dominate execution time.
- **FailureAnalyzer**: Aggregates failure events across rounds to surface recurring error patterns (e.g., repeated timeouts from a specific agent).

These components integrate with the existing audit log and governance systems. The `safeguards/` sub-package complements observability with runtime protection: `RateLimiter` caps LLM call frequency, `TokenBudget` enforces per-task token limits, and `CircuitBreaker` halts calls to an endpoint after repeated failures.

## Dependencies

```bash
pip install -r requirements-dytopo.txt
# or individually:
pip install openai>=1.40 tenacity>=9.0 json-repair>=0.39 httpx>=0.27
pip install pyyaml>=6.0 networkx>=3.0 sentence-transformers>=3.0 pydantic>=2.0
pip install pytest-asyncio>=0.24  # dev/test dependency
```

`sentence-transformers[onnx]` with `onnxruntime` for ONNX INT8 embedding. No PyTorch/CUDA required for embedding. MiniLM-L6-v2 weights (~80 MB) auto-download from HuggingFace on first swarm.
