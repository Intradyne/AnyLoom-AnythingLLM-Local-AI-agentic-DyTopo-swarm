# DyTopo Governance Module - Implementation Summary

> **Note**: The full DyTopo package now lives at `src/dytopo/`. The governance module (`src/dytopo/governance.py`) is integrated directly into the orchestrator's main loop.

## Architecture

The governance module is one of 8 modules in the `src/dytopo/` package:

| Module | Role |
|--------|------|
| `models.py` | Pydantic v2 data models (`AgentMetrics`, `SwarmMetrics`, `SwarmTask`, etc.) |
| `config.py` | YAML configuration with `convergence_threshold` setting |
| `agents.py` | System prompts, JSON schemas, domain agent rosters |
| `router.py` | MiniLM-L6-v2 embedding, cosine similarity, threshold routing |
| `graph.py` | NetworkX DAG, cycle breaking, topological sort |
| `orchestrator.py` | Main loop — calls governance functions after each round |
| **`governance.py`** | **Convergence, stalling, re-delegation (this module)** |
| `audit.py` | JSONL audit logging for all events |

## How Governance Integrates with the Orchestrator

The orchestrator (`run_swarm()`) calls governance functions at specific points:

1. **After each round (t >= 3)**: `detect_convergence()` checks if agent outputs have stabilized
2. **After each round (t >= 2)**: `recommend_redelegation()` identifies stalling or failing agents
3. **On convergence**: Sets `swarm.termination_reason = "convergence"` and breaks early
4. **On re-delegation**: Logs recommendations via `audit.redelegation()` and increments `swarm_metrics.redelegations`

## Functions

### `execute_agent_safe(agent_id, agent_call_coro, timeout_sec=120.0, max_retries=2)`
- Wraps agent execution with timeout and retry
- Returns failure stub on failure (never raises)
- Used by the standalone governance pattern; the orchestrator uses its own `_call_worker()` wrapper

### `detect_convergence(round_history, window_size=3, similarity_threshold=0.95)`
- Compares agent outputs across sliding window using `difflib.SequenceMatcher`
- Returns `(converged: bool, similarity: float)`
- Called by orchestrator after round 3+

### `detect_stalling(agent_id, round_history, window_size=3, similarity_threshold=0.98)`
- Single-agent version of convergence check
- Returns `(stalling: bool, similarity: float)`

### `recommend_redelegation(round_history, agent_roles, stall_threshold=0.98, failure_threshold=2)`
- Identifies stalling, failing, and isolated agents
- Returns `[{agent_id, reason, recommendation, severity}]`
- Called by orchestrator after round 2+

### `update_agent_metrics(agent_state, round_result)`
- Updates dict-based agent state with round results
- Tracks total_rounds, failures, retries, failure_rate

## Metrics Tracked

**Per-agent** (`AgentMetrics` in `models.py`):
- `successful_rounds`, `failed_rounds`, `total_latency_ms`
- `total_tokens_in`, `total_tokens_out`, `times_cited`, `times_isolated`
- Computed: `avg_latency_ms`, `success_rate`, `avg_tokens_per_round`

**Swarm-level** (`SwarmMetrics` in `models.py`):
- `total_rounds`, `total_llm_calls`, `total_tokens`, `total_wall_time_ms`
- `routing_density_per_round`, `convergence_detected_at`
- `agent_failures`, `redelegations`, `per_agent` dict

## Tests

```bash
python tests/test_governance.py
```

## Dependencies

- **Required:** Python 3.9+, `difflib` (stdlib), `asyncio` (stdlib)
- **Optional:** `json-repair>=0.39` (for malformed JSON recovery in `execute_agent_safe`)
