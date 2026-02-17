# DyTopo Integration Example

> **Note**: The full DyTopo package now lives at `src/dytopo/`. This example shows how the orchestrator integrates governance and audit logging.

## Launching a Swarm via MCP

The simplest integration path is through the MCP tools exposed by `qdrant_mcp_server.py`:

```python
# These are MCP tool calls — the server handles everything
result = await swarm_start(
    task="Implement a binary search tree with insert, delete, and search operations",
    domain="code",      # code | math | general
    tau=0.3,            # routing threshold (0.0-1.0)
    k_in=3,             # max incoming messages per agent
    max_rounds=5,       # max reasoning rounds
)
# Returns: "Swarm launched: abc123def456\nDomain: code (...)\nUse swarm_status('abc123def456')..."

status = await swarm_status("abc123def456")
# Returns: "Status: RUNNING (45.2s elapsed)\nRounds completed: 2/5\n..."

result = await swarm_result("abc123def456", include_topology=True)
# Returns: "=== DyTopo Swarm Result ===\nTask: ...\nFinal answer: ...\n=== Topology Log ===\n..."
```

## Programmatic Usage

For direct Python integration without MCP:

```python
import asyncio
from dytopo.models import SwarmDomain, SwarmTask
from dytopo.orchestrator import run_swarm

async def main():
    swarm = SwarmTask(
        task="Prove that the sum of first n odd numbers equals n^2",
        domain=SwarmDomain.MATH,
        tau=0.3,
        K_in=3,
        T_max=5,
    )

    async def on_progress(event: str, data: dict):
        print(f"[{event}] {data}")

    result = await run_swarm(swarm, on_progress=on_progress)

    print(f"Status: {result.status}")
    print(f"Rounds: {len(result.rounds)}")
    print(f"Tokens: {result.total_tokens:,}")
    print(f"Answer: {result.final_answer}")

    # Per-agent metrics
    metrics = result.swarm_metrics
    for agent_id, m in metrics.per_agent.items():
        print(f"  {agent_id}: success={m.success_rate:.0%}, latency={m.avg_latency_ms:.0f}ms")

asyncio.run(main())
```

## Configuration

Override defaults via `dytopo_config.yaml` in the project root:

```yaml
llm:
  base_url: "http://localhost:8008/v1"
  model: "qwen3-30b-a3b-instruct-2507"
  temperature_work: 0.3
  temperature_descriptor: 0.1
routing:
  tau: 0.3
  K_in: 3
orchestration:
  T_max: 5
  convergence_threshold: 0.9
logging:
  log_dir: "~/dytopo-logs"
  save_similarity_matrices: true
```

## Audit Log Analysis

After execution, analyze the JSONL logs:

```bash
# Count events by type
cat ~/dytopo-logs/*/audit.jsonl | jq -r '.event_type' | sort | uniq -c

# Find all failures
cat ~/dytopo-logs/*/audit.jsonl | jq 'select(.event_type == "agent_failed")'

# Find convergence points
cat ~/dytopo-logs/*/audit.jsonl | jq 'select(.event_type == "convergence_detected")'
```

## Package Structure

See `docs/dytopo-swarm.md` for the full architecture documentation.
