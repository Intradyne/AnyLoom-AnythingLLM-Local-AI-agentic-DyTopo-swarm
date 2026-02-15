# DyTopo Governance Module (Standalone)

> **Note**: This directory (`dytopo/`) contains the standalone governance module with `governance.py`, `audit.py`, and `models.py`. The **full DyTopo package** lives at `src/dytopo/` and includes all 8 modules (models, config, agents, router, graph, orchestrator, governance, audit). The MCP server imports from `src/dytopo/`.

## Governance Functions

The governance module (`governance.py`) provides failure-resilient agent execution, convergence detection, and adaptive re-delegation:

### `execute_agent_safe(agent_id, agent_call_coro, timeout_sec, max_retries)`
Wraps agent LLM calls with timeout protection, retry logic, and graceful degradation. Returns a failure stub instead of raising exceptions.

### `detect_convergence(round_history, window_size, similarity_threshold)`
Compares recent rounds using `difflib.SequenceMatcher`. Returns `(converged: bool, similarity: float)`.

### `detect_stalling(agent_id, round_history, window_size, similarity_threshold)`
Monitors a single agent for repeated outputs across rounds. Returns `(stalling: bool, similarity: float)`.

### `recommend_redelegation(round_history, agent_roles, stall_threshold, failure_threshold)`
Analyzes performance and suggests modifications. Returns list of `{agent_id, reason, recommendation, severity}`.

### `update_agent_metrics(agent_state, round_result)`
Updates agent state dict with execution metrics (total rounds, failures, retries, failure rate).

## Audit Logging

`SwarmAuditLog` writes JSONL events to `~/dytopo-logs/{task_id}/audit.jsonl`:
- `swarm_started`, `round_started`, `agent_executed`, `agent_failed`
- `convergence_detected`, `redelegation`, `swarm_completed`

## Full Package

For the complete DyTopo implementation including routing, orchestration, and MCP tools, see:
- Package: `src/dytopo/`
- Docs: `docs/dytopo-swarm.md`
- Config: `dytopo_config.yaml`
- Dependencies: `requirements-dytopo.txt`
