"""
Living Documentation Generator
===============================

Auto-generates documentation from code execution data.

Generates:
1. overview-obsidian.canvas - Visual system map
2. API reference from docstrings
3. Architecture diagrams
4. Troubleshooting guides
5. Performance reports

Usage:
    # Generate overview canvas
    await DocumentationGenerator.generate_overview_canvas(
        Path("docs/overview-obsidian.canvas")
    )

    # Generate API reference
    await DocumentationGenerator.generate_api_reference(
        Path("docs/api-reference.md"),
        orchestrator_module
    )
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dytopo.documentation.generator")


class DocumentationGenerator:
    """
    Auto-generate documentation from execution data.

    Provides methods to generate:
    - Obsidian canvas files (visual system maps)
    - API reference documentation
    - Architecture diagrams
    - Troubleshooting guides
    """

    @staticmethod
    async def generate_overview_canvas(
        output_path: Path,
        include_metrics: bool = True,
    ):
        """
        Generate overview-obsidian.canvas with system architecture.

        Creates a visual map of the DyTopo system showing:
        - Core components (orchestrator, routing, agents)
        - Data flow patterns
        - Observability layer
        - Integration points

        Args:
            output_path: Where to write the canvas file
            include_metrics: Whether to include metrics/observability nodes
        """
        nodes = []
        edges = []

        # Calculate positions in a grid
        center_x, center_y = 0, 0
        spacing = 350

        # Central orchestrator node
        nodes.append({
            "id": "orchestrator",
            "type": "text",
            "text": (
                "## Orchestrator\n\n"
                "**run_swarm(task, domain, tau, K_in, T_max)**\n\n"
                "Core multi-agent swarm execution:\n"
                "- Round-based iteration\n"
                "- Manager-driven coordination\n"
                "- Dynamic routing between rounds\n"
                "- Convergence detection\n"
                "- Graceful termination\n\n"
                "📍 [orchestrator.py](../src/dytopo/orchestrator.py)"
            ),
            "x": center_x,
            "y": center_y,
            "width": 320,
            "height": 280,
            "color": "4"
        })

        # Manager Agent
        nodes.append({
            "id": "manager",
            "type": "text",
            "text": (
                "## Manager Agent\n\n"
                "Sets goals and decides termination:\n"
                "- Analyzes round history\n"
                "- Generates round goals\n"
                "- Determines when task complete\n"
                "- Extracts final answer\n\n"
                "📍 [agents.py](../src/dytopo/agents.py)"
            ),
            "x": center_x - spacing,
            "y": center_y - spacing,
            "width": 280,
            "height": 220,
            "color": "3"
        })

        # Worker Agents
        nodes.append({
            "id": "workers",
            "type": "text",
            "text": (
                "## Worker Agents\n\n"
                "Domain-specific specialist agents:\n"
                "- Developer\n"
                "- Researcher\n"
                "- Problem Parser\n"
                "- Solver\n"
                "- Verifier\n"
                "- Critic\n"
                "- Synthesizer\n\n"
                "Each generates key/query/work\n\n"
                "📍 [agents.py](../src/dytopo/agents.py)"
            ),
            "x": center_x + spacing,
            "y": center_y - spacing,
            "width": 280,
            "height": 260,
            "color": "2"
        })

        # Routing Engine
        nodes.append({
            "id": "routing",
            "type": "text",
            "text": (
                "## Routing Engine\n\n"
                "Semantic message routing:\n"
                "- Descriptor embedding (key/query)\n"
                "- Cosine similarity matching\n"
                "- DAG construction\n"
                "- Cycle detection & breaking\n"
                "- Topological execution order\n\n"
                "📍 [router.py](../src/dytopo/router.py)\n"
                "📍 [graph.py](../src/dytopo/graph.py)"
            ),
            "x": center_x - spacing,
            "y": center_y + spacing,
            "width": 300,
            "height": 260,
            "color": "5"
        })

        # Governance
        nodes.append({
            "id": "governance",
            "type": "text",
            "text": (
                "## Governance Layer\n\n"
                "Failure resilience & adaptation:\n"
                "- Convergence detection\n"
                "- Stalling detection\n"
                "- Re-delegation recommendations\n"
                "- Graceful failure handling\n"
                "- Agent metrics tracking\n\n"
                "📍 [governance.py](../src/dytopo/governance.py)"
            ),
            "x": center_x + spacing,
            "y": center_y + spacing,
            "width": 300,
            "height": 240,
            "color": "1"
        })

        if include_metrics:
            # Observability Layer
            nodes.append({
                "id": "observability",
                "type": "text",
                "text": (
                    "## Observability Layer\n\n"
                    "Distributed tracing & metrics:\n"
                    "- Trace propagation\n"
                    "- Span collection\n"
                    "- Latency metrics (p95, p99)\n"
                    "- Failure tracking\n"
                    "- Performance profiling\n"
                    "- Bottleneck analysis\n\n"
                    "📍 [observability/](../src/dytopo/observability/)"
                ),
                "x": center_x,
                "y": center_y + spacing * 2,
                "width": 320,
                "height": 260,
                "color": "6"
            })

            # Audit Log
            nodes.append({
                "id": "audit",
                "type": "text",
                "text": (
                    "## Audit Log\n\n"
                    "JSONL event logging:\n"
                    "- Swarm lifecycle events\n"
                    "- Agent execution tracking\n"
                    "- Routing decisions\n"
                    "- Convergence events\n"
                    "- Failure records\n\n"
                    "📍 [audit.py](../src/dytopo/audit.py)"
                ),
                "x": center_x - spacing * 1.5,
                "y": center_y + spacing * 2,
                "width": 280,
                "height": 220,
                "color": "6"
            })

        # Edges showing relationships
        edges.append({
            "id": "edge1",
            "fromNode": "manager",
            "toNode": "orchestrator",
            "label": "sets goals"
        })

        edges.append({
            "id": "edge2",
            "fromNode": "orchestrator",
            "toNode": "workers",
            "label": "coordinates"
        })

        edges.append({
            "id": "edge3",
            "fromNode": "orchestrator",
            "toNode": "routing",
            "label": "builds graph"
        })

        edges.append({
            "id": "edge4",
            "fromNode": "routing",
            "toNode": "workers",
            "label": "routes messages"
        })

        edges.append({
            "id": "edge5",
            "fromNode": "governance",
            "toNode": "orchestrator",
            "label": "monitors"
        })

        if include_metrics:
            edges.append({
                "id": "edge6",
                "fromNode": "observability",
                "toNode": "orchestrator",
                "label": "instruments"
            })

            edges.append({
                "id": "edge7",
                "fromNode": "audit",
                "toNode": "orchestrator",
                "label": "logs events"
            })

        # Build canvas JSON
        canvas = {
            "nodes": nodes,
            "edges": edges
        }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write canvas file
        output_path.write_text(
            json.dumps(canvas, indent=2),
            encoding="utf-8"
        )

        logger.info(
            f"Generated overview canvas: {output_path} "
            f"({len(nodes)} nodes, {len(edges)} edges)"
        )

    @staticmethod
    async def generate_api_reference(
        output_path: Path,
        *modules,
    ):
        """
        Generate API reference from module docstrings.

        Args:
            output_path: Where to write the markdown file
            *modules: Python modules to document
        """
        lines = [
            "# DyTopo API Reference",
            "",
            "Auto-generated API documentation for the DyTopo multi-agent swarm system.",
            "",
            "## Table of Contents",
            "",
        ]

        # Build table of contents
        for module in modules:
            module_name = module.__name__.split(".")[-1]
            lines.append(f"- [{module_name}](#{module_name})")

        lines.append("")

        # Document each module
        for module in modules:
            module_name = module.__name__.split(".")[-1]

            lines.extend([
                "---",
                "",
                f"## {module_name}",
                "",
            ])

            # Module docstring
            module_doc = inspect.getdoc(module)
            if module_doc:
                lines.append(module_doc)
                lines.append("")

            # Document classes
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.startswith("_"):
                    continue

                lines.append(f"### `{name}`")
                lines.append("")

                class_doc = inspect.getdoc(obj)
                if class_doc:
                    lines.append(class_doc)
                    lines.append("")

                # Document methods
                methods = inspect.getmembers(obj, inspect.isfunction)
                public_methods = [
                    (n, m) for n, m in methods if not n.startswith("_")
                ]

                if public_methods:
                    lines.append("**Methods:**")
                    lines.append("")

                    for method_name, method in public_methods:
                        sig = inspect.signature(method)
                        lines.append(f"#### `{method_name}{sig}`")
                        lines.append("")

                        method_doc = inspect.getdoc(method)
                        if method_doc:
                            lines.append(method_doc)
                        lines.append("")

            # Document functions
            for name, obj in inspect.getmembers(module, inspect.isfunction):
                if name.startswith("_"):
                    continue

                sig = inspect.signature(obj)
                lines.append(f"### `{name}{sig}`")
                lines.append("")

                func_doc = inspect.getdoc(obj)
                if func_doc:
                    lines.append(func_doc)
                lines.append("")

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write markdown file
        output_path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

        logger.info(f"Generated API reference: {output_path}")

    @staticmethod
    async def generate_troubleshooting_guide(output_path: Path):
        """
        Generate troubleshooting guide for common issues.

        Args:
            output_path: Where to write the markdown file
        """
        content = """# DyTopo Troubleshooting Guide

## Common Issues and Solutions

### Issue: Swarm Takes Too Long (>5 minutes)

**Symptoms:**
- Swarm execution exceeds expected time
- Some rounds take significantly longer than others

**Diagnosis:**
1. Check trace analysis for bottlenecks:
   ```python
   from dytopo.observability import TraceCollector
   analysis = await TraceCollector.analyze_trace(trace_id)
   print(analysis["slowest_operations"])
   ```

2. Review performance score:
   ```python
   from dytopo.observability import BottleneckAnalyzer
   report = await BottleneckAnalyzer.analyze(trace_id)
   print(f"Score: {report['performance_score']}/100")
   ```

**Solutions:**
- If parallelization_factor < 2.0: Increase concurrent agent execution
- If long-pole operations detected: Optimize slow agents or reduce max_tokens
- If routing takes >1s: Reduce agent count or simplify descriptors

---

### Issue: Agent Failures / Timeouts

**Symptoms:**
- Agents repeatedly fail or timeout
- Error rate >10%

**Diagnosis:**
```python
from dytopo.observability import failure_analyzer
patterns = await failure_analyzer.analyze_patterns()
print(patterns["most_common_error"])
```

**Solutions:**
- **TimeoutError**: Increase timeout in config, reduce max_tokens
- **JSONDecodeError**: Enable JSON repair, review LLM response format
- **ConnectionError**: Check LM Studio/vLLM is running, verify base_url

---

### Issue: Convergence Detected Too Early

**Symptoms:**
- Swarm terminates before task is complete
- Final answer is incomplete or wrong

**Diagnosis:**
Check convergence settings in config:
```yaml
orchestration:
  convergence_threshold: 0.95  # Increase to 0.98 for stricter
```

**Solutions:**
- Increase convergence_threshold (0.95 → 0.98)
- Increase convergence window_size (3 → 4 rounds)
- Add diversity to agent prompts to encourage exploration

---

### Issue: Routing Graph Is Too Sparse/Dense

**Symptoms:**
- Too sparse: Agents isolated, missing relevant information
- Too dense: Information overload, no filtering benefit

**Diagnosis:**
```python
# Check routing density per round
for round in swarm.rounds:
    print(f"Round {round.round_num}: {round.routing_stats['density']:.2%}")
```

**Solutions:**
- **Too sparse** (density <10%): Decrease tau (0.3 → 0.2)
- **Too dense** (density >70%): Increase tau (0.3 → 0.4) or decrease K_in

---

### Issue: High Token Consumption

**Symptoms:**
- total_tokens exceeds budget
- LLM API costs too high

**Diagnosis:**
```python
from dytopo.observability import metrics
stats = await metrics.get_stats("llm_tokens_total")
print(f"Average tokens/call: {stats['mean']}")
```

**Solutions:**
- Reduce max_tokens_work in config (4096 → 2048)
- Enable work truncation in routing
- Reduce T_max (5 → 3 rounds)
- Use broadcast_round_1: false to skip round 1 full broadcast

---

### Issue: Memory Issues / OOM

**Symptoms:**
- Process killed by OOM
- Memory usage grows unbounded

**Diagnosis:**
```python
import psutil
process = psutil.Process()
print(f"Memory: {process.memory_info().rss / 1024**2:.0f} MB")
```

**Solutions:**
- Enable metric cleanup:
  ```python
  await metrics.cleanup_old_metrics()
  ```
- Clear trace collector periodically:
  ```python
  await TraceCollector.clear()
  ```
- Reduce retention_hours for metrics (24 → 6)

---

## Performance Optimization Checklist

- [ ] Run performance profiler to identify bottlenecks
- [ ] Check parallelization factor (target: >2.0)
- [ ] Review slowest operations in trace
- [ ] Optimize routing density (target: 20-50%)
- [ ] Enable convergence detection
- [ ] Tune tau and K_in for routing
- [ ] Reduce max_tokens if appropriate
- [ ] Monitor failure patterns and address root causes

---

## Debugging Tools

### View Trace
```python
from dytopo.observability import TraceCollector
trace_id = trace_id_var.get()
await TraceCollector.export_trace(trace_id, Path("trace.json"))
```

### View Metrics
```python
from dytopo.observability import metrics
summary = await metrics.get_summary()
print(summary)
```

### View Failures
```python
from dytopo.observability import failure_analyzer
patterns = await failure_analyzer.analyze_patterns()
print(patterns)
```

### Generate Performance Report
```python
from dytopo.observability import BottleneckAnalyzer
report = await BottleneckAnalyzer.analyze(trace_id)
print(BottleneckAnalyzer.format_report(report))
```

---

## Getting Help

If you encounter an issue not covered here:

1. Export your trace: `await TraceCollector.export_trace(trace_id, Path("debug.json"))`
2. Export failures: `await failure_analyzer.export_failures_json(Path("failures.json"))`
3. Check audit log: `~/dytopo-logs/{task_id}/audit.jsonl`
4. Open an issue with the exported data
"""

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write guide
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"Generated troubleshooting guide: {output_path}")

    @staticmethod
    async def generate_architecture_doc(output_path: Path):
        """
        Generate architecture documentation.

        Args:
            output_path: Where to write the markdown file
        """
        content = """# DyTopo Architecture

## System Overview

DyTopo is a dynamic topology multi-agent swarm system with semantic routing between agents.

### Key Components

1. **Orchestrator** (`orchestrator.py`)
   - Main execution loop
   - Round-based iteration
   - Manager coordination
   - Convergence detection

2. **Routing Engine** (`router.py`, `graph.py`)
   - Semantic descriptor matching
   - DAG construction
   - Topological execution order

3. **Governance** (`governance.py`)
   - Convergence detection
   - Stalling detection
   - Re-delegation recommendations

4. **Observability** (`observability/`)
   - Distributed tracing
   - Metrics collection
   - Performance profiling
   - Failure analysis

5. **Audit Logging** (`audit.py`)
   - JSONL event stream
   - Execution tracking

---

## Execution Flow

```
User Task
    ↓
Orchestrator.run_swarm()
    ↓
    ├─ Round 1: BROADCAST (all agents see all outputs)
    │   ├─ Manager sets goal
    │   ├─ All agents execute in parallel
    │   └─ No routing (full mesh)
    │
    ├─ Round 2+: SEMANTIC ROUTING
    │   ├─ Manager sets goal
    │   ├─ Phase A: Generate descriptors (key/query)
    │   ├─ Phase B: Build routing graph
    │   │   └─ Embed descriptors → similarity → DAG
    │   └─ Phase C: Execute agents in topological order
    │       └─ Each agent receives routed messages
    │
    ├─ Convergence Check
    │   └─ If converged → terminate
    │
    └─ Max Rounds Check
        └─ If T_max reached → extract answer

Final Answer
```

---

## Routing Algorithm

1. **Descriptor Generation**
   - Each agent produces (key, query)
   - key = "what I offer"
   - query = "what I need"

2. **Embedding**
   - Embed all keys and queries using sentence-transformers

3. **Similarity Matching**
   - For each agent A:
     - Compare A.query to all other agents' keys
     - Add edge B → A if similarity(A.query, B.key) > tau

4. **DAG Construction**
   - Detect cycles using topological sort
   - Break cycles by removing lowest-weight edge

5. **Execution Order**
   - Topological sort of DAG
   - Agents with no dependencies execute first
   - Messages flow along edges

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

### Performance Profiling

- **CPU Profiling**: cProfile integration for hot path identification
- **Async Profiling**: Trace-based timing analysis
- **Bottleneck Detection**: Identifies long-pole operations

---

## Configuration

See `dytopo_config.yaml`:

```yaml
llm:
  base_url: "http://localhost:1234/v1"
  model: "qwen3-30b-a3b-instruct-2507"
  timeout_seconds: 120

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

| Metric | Typical Value |
|--------|---------------|
| Round 1 (broadcast) | 10-20s |
| Round 2+ (routed) | 8-15s |
| Routing computation | <1s |
| Descriptor generation | 2-5s |
| Total swarm (5 rounds) | 45-75s |
| Parallelization factor | 2.0-3.5x |

---

## References

- Paper: arXiv 2602.06039 (Lu et al., Feb 2026)
- Code: [src/dytopo/](../src/dytopo/)
- Docs: [docs/](../docs/)
"""

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write doc
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"Generated architecture doc: {output_path}")
