# DyTopo Governance Module - Implementation Summary

> **Note**: The full DyTopo package now lives at `src/dytopo/`. The governance module (`src/dytopo/governance.py`) is integrated directly into the orchestrator's main loop.

## Architecture

The governance module is one of 12 core modules in the `src/dytopo/` package (which also includes sub-packages for observability, safeguards, messaging, routing, delegation, and documentation):

| Module | Role |
|--------|------|
| `models.py` | Pydantic v2 data models (`AgentMetrics`, `SwarmMetrics`, `SwarmTask`, etc.) |
| `config.py` | YAML configuration with `convergence_threshold`, `checkpoint`, and `verification` settings |
| `agents.py` | System prompts, JSON schemas, domain agent rosters |
| `router.py` | MiniLM-L6-v2 embedding, cosine similarity, threshold routing, intent embedding enrichment |
| `stigmergic_router.py` | Trace-aware topology: Qdrant-persisted swarm traces, time-decayed boost matrix |
| `graph.py` | NetworkX DAG, cycle breaking, topological sort |
| `orchestrator.py` | Main loop — calls governance functions after each round; integrates checkpoint, policy, verifier, stalemate via guarded imports |
| **`governance.py`** | **Convergence, stalling, re-delegation, Aegean consensus, stalemate detection, generalist fallback (this module)** |
| `checkpoint.py` | CheckpointManager — atomic crash-recovery persistence of SwarmTask state |
| `policy.py` | PolicyEnforcer (PCAS-Lite) — deny-first tool-call policy enforcement |
| `verifier.py` | OutputVerifier — deterministic output verification (syntax, schema, no LLM) |
| `audit.py` | JSONL audit logging for all events |

## How Governance Integrates with the Orchestrator

The orchestrator (`run_swarm()`) calls governance functions at specific points:

1. **After each round (t >= 3)**: `detect_convergence()` checks if agent outputs have stabilized
2. **After each round (t >= 2)**: `check_aegean_termination()` runs embedding-based consensus vote
3. **After each round (t >= 2)**: `recommend_redelegation()` identifies stalling or failing agents
4. **After governance checks**: `StalemateDetector.check()` analyzes routing patterns for ping-pong, no-progress, and regression stalemates (when `_HAS_STALEMATE` is True)
5. **On convergence**: Sets `swarm.termination_reason = "convergence"` and breaks early
6. **On Aegean consensus**: Sets `swarm.termination_reason = "aegean_consensus"` and breaks early
7. **On stalemate**: Injects generalist fallback agent via `get_generalist_fallback_agent()`, or triggers forced termination
8. **On re-delegation**: Logs recommendations via `audit.redelegation()` and increments `swarm_metrics.redelegations`
9. **After each step**: Saves checkpoint via `CheckpointManager.save()` (when `_HAS_CHECKPOINT` is True)
10. **Before tool execution**: Checks `PolicyEnforcer.enforce()` for deny-first policy gating (when `_HAS_POLICY` is True)
11. **After agent execution**: Runs `OutputVerifier.verify()` for deterministic output verification (when `_HAS_VERIFIER` is True)

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

### `compute_consensus_matrix(agent_outputs, model)`
- Embeds all agent outputs via MiniLM-L6-v2 and computes pairwise cosine similarity
- Returns NxN similarity matrix used by Aegean termination check
- Non-fatal — errors return None

### `check_aegean_termination(agent_outputs, consensus_threshold=0.85, vote_threshold=0.75)`
- Embedding-based consensus check across agent outputs
- Each agent "votes" to terminate if its avg similarity to others exceeds `consensus_threshold`
- If >= `vote_threshold` (75%) of agents vote, the swarm terminates early
- Runs after convergence detection in rounds >= 2
- Returns `(should_terminate: bool, votes: dict, avg_similarity: float)`

### `update_agent_metrics(agent_state, round_result)`
- Updates dict-based agent state with round results
- Tracks total_rounds, failures, retries, failure_rate

### `StalemateDetector` (class)
- Detects systemic stalemates in the swarm routing topology across rounds
- Three detection patterns: **ping-pong** (two agents routing to each other without involving others), **no progress** (convergence score unchanged for multiple rounds), **regression** (convergence score actively decreasing)
- `check(round_history, convergence_scores)` returns a `StalemateResult`

### `StalemateResult` (dataclass)
- `is_stalled: bool`, `reason: str`, `suggested_action: str` (one of `"generalist"`, `"force_terminate"`, `"human_in_loop"`, `""`), `stale_pair: tuple[str, str] | None`

### `get_generalist_fallback_agent(agent_states, stale_pair)`
- Selects a fallback agent to break a stalemate
- Picks the agent NOT in the stale pair that has been used least (fewest rounds participated)
- Returns the agent_id string, or None if no suitable fallback exists

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
