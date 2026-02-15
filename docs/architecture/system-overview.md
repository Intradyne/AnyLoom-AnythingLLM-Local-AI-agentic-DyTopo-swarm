# DyTopo Architecture

## System Overview

DyTopo is a dynamic topology multi-agent swarm system with semantic routing between agents, async parallel execution, and comprehensive observability.

### Core Modules

1. **Orchestrator** (`orchestrator.py`)
   - Async parallel swarm execution loop
   - Backend-agnostic LLM client (LM Studio or vLLM)
   - Semaphore-controlled concurrency (1 for LM Studio, 8 for vLLM)
   - Round 1: parallel broadcast via `asyncio.gather()`
   - Rounds 2+: tiered parallel execution within topological tiers
   - Manager-driven coordination and convergence detection

2. **Routing Engine** (`router.py`, `graph.py`, `routing/async_engine.py`)
   - Semantic descriptor matching via MiniLM-L6-v2 embeddings
   - DAG construction with cycle breaking
   - Topological tier computation (`get_execution_tiers()`) for parallel-within-tier execution
   - AsyncRoutingEngine with lock-based embedding serialization

3. **Governance** (`governance.py`)
   - Convergence detection
   - Stalling detection
   - Re-delegation recommendations
   - Agent metrics tracking

4. **Observability** (`observability/`)
   - Distributed tracing (TraceContext, contextvars propagation)
   - Metrics collection (percentiles p50/p95/p99, Prometheus export)
   - Performance profiling and bottleneck analysis
   - Failure analysis and pattern detection

5. **Safeguards** (`safeguards/`)
   - RateLimiter (token bucket)
   - TokenBudget (hard spending cap)
   - CircuitBreaker (fail-fast on repeated errors)

6. **Messaging** (`messaging/`)
   - AgentMessage (typed structured payloads)
   - MessageHistory (per-agent, prunable)
   - MessageRouter (graph-aware delivery)

7. **Delegation** (`delegation/`)
   - DelegationManager (subtask spawning)
   - Depth-limited nesting, concurrency control, timeout protection
   - Lineage tracking via DelegationRecord

8. **Audit Logging** (`audit.py`)
   - JSONL event stream to `~/dytopo-logs/{task_id}/audit.jsonl`
   - Swarm lifecycle, agent execution, routing, convergence events

---

## Execution Flow

```
User Task
    |
Orchestrator.run_swarm()
    |
    +-- Round 1: BROADCAST
    |   +-- Manager sets goal
    |   +-- All agents execute in parallel (asyncio.gather)
    |   +-- No routing (full mesh)
    |
    +-- Round 2+: SEMANTIC ROUTING
    |   +-- Manager sets goal
    |   +-- Phase A: Generate descriptors (parallel, asyncio.gather)
    |   +-- Phase B: Build routing graph (CPU-only, ~10-50ms)
    |   |   +-- Embed descriptors -> similarity -> threshold -> DAG
    |   +-- Phase C: Execute agents in topological tiers
    |       +-- Tier 0: no-dependency agents (parallel)
    |       +-- Tier 1: depends on tier 0 (parallel within tier)
    |       +-- ... complete each tier before starting next
    |
    +-- Convergence Check
    |   +-- If converged -> terminate
    |
    +-- Max Rounds Check
        +-- If T_max reached -> extract answer

Final Answer
```

---

## Routing Algorithm

1. **Descriptor Generation**
   - Each agent produces (key, query)
   - key = "what I offer"
   - query = "what I need"

2. **Embedding**
   - Embed all keys and queries using sentence-transformers (MiniLM-L6-v2)

3. **Similarity Matching**
   - For each agent A:
     - Compare A.query to all other agents' keys
     - Add edge B -> A if similarity(A.query, B.key) > tau

4. **DAG Construction**
   - Detect cycles using topological sort
   - Break cycles by removing lowest-weight edge

5. **Tier Computation**
   - `get_execution_tiers()` via `nx.topological_generations()`
   - Agents in the same tier have no inter-dependencies
   - Tier 0: in-degree 0 (sources), Tier N: depends only on tiers < N

---

## Concurrency Model

```
                        +-- lmstudio (port 1234) --+
Config backend ----+--->|  max_concurrent = 1       |---> Sequential
                   |    +---------------------------+
                   |
                   |    +-- vllm (port 8000) -------+
                   +--->|  max_concurrent = 8       |---> Parallel
                        +---------------------------+

Semaphore wraps every _llm_call():
  async with semaphore:
      response = await client.chat.completions.create(...)
```

With LM Studio (max_concurrent=1), the semaphore serializes all calls — behavior identical to sequential execution. With vLLM (max_concurrent=8), up to 8 LLM calls run concurrently within `asyncio.gather()`.

---

## Observability Architecture

### Distributed Tracing

- **Trace ID**: Unique per swarm execution
- **Spans**: Hierarchical units of work (round, agent_call, routing, etc.)
- **Propagation**: Via Python contextvars across async calls

### Metrics Collection

- **Latency**: LLM call duration, routing time, agent execution time
- **Throughput**: Tokens/sec, requests/sec
- **Errors**: Failure counts by type, error rates
- **Resources**: Token consumption, memory usage
- **Percentiles**: p50, p95, p99 via MetricsCollector
- **Export**: Prometheus text format

### Performance Profiling

- **Bottleneck Detection**: Identifies long-pole operations
- **Parallelization Factor**: Measures effective concurrency utilization
- **Recommendations**: Actionable optimization suggestions

---

## Configuration

See `dytopo_config.yaml`:

```yaml
llm:
  base_url: "http://localhost:1234/v1"
  model: "qwen3-30b-a3b-instruct-2507"

routing:
  tau: 0.3              # Similarity threshold
  K_in: 3               # Max incoming edges
  broadcast_round_1: true

orchestration:
  T_max: 5              # Max rounds
  convergence_threshold: 0.95
  fallback_on_isolation: true

logging:
  log_dir: "~/dytopo-logs"
  save_similarity_matrices: true

concurrency:
  backend: "lmstudio"   # "lmstudio" or "vllm"
  max_concurrent: 1     # 1 for lmstudio, 8 for vllm
  # vllm_base_url: "http://localhost:8000/v1"
```

---

## Extension Points

### Custom Agents

Add new agent roles in `agents.py`:

```python
class AgentRole(str, Enum):
    # ... existing roles
    YOUR_CUSTOM_ROLE = "your_custom_role"

# Add system prompt
SYSTEM_PROMPTS = {
    AgentRole.YOUR_CUSTOM_ROLE: "You are a custom agent...",
}
```

### Custom Routing

Extend `router.py` to implement alternative routing strategies:

```python
def custom_routing_strategy(descriptors, tau, K_in):
    # Your logic here
    return edges
```

### Custom Metrics

Add domain-specific metrics:

```python
from dytopo.observability import metrics

await metrics.record(
    "custom_metric_name",
    value,
    label1="value1",
    label2="value2"
)
```

---

## Performance Characteristics

| Metric | Sequential (LM Studio) | Parallel (vLLM) |
|--------|----------------------|-----------------|
| Round 1 (broadcast, 4 agents) | 40-80s | 10-20s |
| Round 2+ (routed, 4 agents) | 40-60s | 12-20s |
| Descriptor generation (Phase A) | 12-20s | 3-5s |
| Routing computation (Phase B) | <1s | <1s |
| Total swarm (5 rounds) | 3-8 min | 1-3 min |

---

## Project Structure

```
src/dytopo/
  models.py          # Pydantic v2 data models
  config.py          # YAML config loader
  agents.py          # System prompts, schemas, roster builder
  router.py          # MiniLM embedding, similarity, threshold
  graph.py           # DAG construction, cycle breaking, tiers
  orchestrator.py    # Async parallel swarm execution loop
  governance.py      # Convergence, stalling, re-delegation
  audit.py           # JSONL event logging
  observability/     # Tracing, metrics, profiling, failures
  safeguards/        # Rate limiter, token budget, circuit breaker
  messaging/         # Typed message passing
  routing/           # AsyncRoutingEngine
  delegation/        # Subtask delegation manager
  documentation/     # Auto-generated docs
```

---

## References

- Paper: arXiv 2602.06039 (Lu et al., Feb 2026)
- Code: [src/dytopo/](../src/dytopo/)
- Docs: [docs/](../docs/)
