# DyTopo Observability Layer

## Overview

The DyTopo observability layer provides comprehensive monitoring, tracing, and performance analysis for the multi-agent swarm system. This document describes the architecture, features, and usage of the observability infrastructure.

## Components

### 1. Distributed Tracing (`observability/tracing.py`)

Tracks execution flow across async operations with hierarchical span tracking.

**Features:**
- Automatic trace ID generation and propagation
- Parent-child span relationships
- Error capture with full context
- JSON export (Jaeger/Zipkin compatible)

**Usage:**
```python
from dytopo.observability import TraceContext, TraceCollector, trace_id_var

# Wrap operations with trace context
async with TraceContext("swarm_execution", task="prove theorem"):
    for round_num in range(T_max):
        async with TraceContext("round", round_num=round_num):
            # Your code here
            pass

# Export trace
trace_id = trace_id_var.get()
await TraceCollector.export_trace(trace_id, Path("traces/trace.json"))

# Analyze trace
analysis = await TraceCollector.analyze_trace(trace_id)
print(f"Total duration: {analysis['total_duration']:.2f}s")
print(f"Error rate: {analysis['error_rate']:.1%}")
```

### 2. Enhanced Metrics (`observability/metrics.py`)

Collects metrics with statistical aggregation and percentile calculations.

**Features:**
- Label-based filtering
- Percentile calculation (p50, p95, p99)
- Prometheus text format export
- Automatic retention management

**Usage:**
```python
from dytopo.observability import metrics

# Record metrics
await metrics.record(
    "llm_call_duration_seconds",
    1.234,
    agent="developer",
    status="success"
)

# Get statistics
stats = await metrics.get_stats(
    "llm_call_duration_seconds",
    agent="developer"
)
print(f"p95 latency: {stats['p95']:.3f}s")

# Export to Prometheus
await metrics.export_prometheus(Path("metrics.prom"))
```

### 3. Performance Profiling (`observability/profiling.py`)

Identifies bottlenecks and generates optimization recommendations.

**Features:**
- CPU profiling (cProfile integration)
- Async operation timing
- Bottleneck detection
- Performance score calculation
- Actionable recommendations

**Usage:**
```python
from dytopo.observability import BottleneckAnalyzer

# Analyze a trace
analysis = await BottleneckAnalyzer.analyze(trace_id)

print(f"Performance score: {analysis['performance_score']}/100")
print(f"Parallelization: {analysis['parallelization_factor']:.2f}x")

# Get formatted report
report = BottleneckAnalyzer.format_report(analysis)
print(report)
```

### 4. Failure Analysis (`observability/failures.py`)

Tracks failures and suggests recovery strategies.

**Features:**
- Detailed failure event tracking
- Pattern detection (temporal clustering)
- Recovery strategy suggestions
- Failure rate monitoring

**Usage:**
```python
from dytopo.observability import failure_analyzer

# Record failure
failure_id = await failure_analyzer.record_failure(
    component="orchestrator",
    operation="agent_call",
    error=TimeoutError("Agent timeout"),
    agent_id="developer"
)

# Mark as recovered
await failure_analyzer.mark_recovered(failure_id, "retry")

# Analyze patterns
patterns = await failure_analyzer.analyze_patterns()
print(f"Total failures: {patterns['total_failures']}")
print(f"Recovery rate: {patterns['recovery_rate']:.1%}")
```

## Integration with Existing Code

### Instrumenting the Orchestrator

Add tracing to the main orchestration loop:

```python
from dytopo.observability import TraceContext, metrics, failure_analyzer

async def run_swarm(swarm: SwarmTask, ...):
    async with TraceContext("swarm_execution", task=swarm.task):
        for t in range(1, swarm.T_max + 1):
            async with TraceContext("round", round_num=t):

                # Manager call with tracing
                async with TraceContext("manager_call"):
                    start = time.time()
                    decision, tokens = await _call_manager(...)

                    # Record metrics
                    await metrics.record(
                        "llm_call_duration_seconds",
                        time.time() - start,
                        agent="manager",
                        status="success"
                    )

                # Routing with tracing
                async with TraceContext("routing"):
                    graph = await build_routing_graph(...)

                # Agent execution
                async with TraceContext("agent_execution"):
                    for agent_id in execution_order:
                        try:
                            async with TraceContext("agent_call", agent_id=agent_id):
                                result = await _call_worker(...)
                        except Exception as e:
                            await failure_analyzer.record_failure(
                                "orchestrator",
                                "agent_call",
                                e,
                                agent_id=agent_id,
                                round=t
                            )
                            raise
```

## Performance Budget

The observability layer is designed with minimal overhead:

- **Tracing**: <5% latency increase
- **Metrics**: <2% latency increase
- **Total overhead**: <10% latency increase

These are acceptable costs for production observability.

## Export Formats

### Trace Export (JSON)

```json
{
  "trace_id": "abc123def456",
  "total_spans": 45,
  "start_time": 1234567890.123,
  "end_time": 1234567895.456,
  "total_duration": 5.333,
  "spans": [
    {
      "span_id": "span001",
      "parent_id": null,
      "trace_id": "abc123def456",
      "operation": "swarm_execution",
      "start_time": 1234567890.123,
      "end_time": 1234567895.456,
      "duration": 5.333,
      "status": "success",
      "metadata": {"task": "prove theorem"}
    }
  ]
}
```

### Metrics Export (Prometheus)

```
# TYPE llm_call_duration_seconds gauge
llm_call_duration_seconds{agent="developer",status="success"} 1.234 1234567890000
llm_call_duration_seconds{agent="manager",status="success"} 0.567 1234567891000

# TYPE llm_tokens_total gauge
llm_tokens_total{agent="developer",token_type="completion"} 1500 1234567890000
```

## Visualization

### Using with Jaeger

1. Export trace to JSON:
   ```python
   await TraceCollector.export_trace(trace_id, Path("trace.json"))
   ```

2. Convert to Jaeger format (custom script needed)

3. Import into Jaeger UI

### Using with Grafana

1. Export metrics to Prometheus:
   ```python
   await metrics.export_prometheus(Path("metrics.prom"))
   ```

2. Configure Prometheus to scrape the file

3. Create Grafana dashboard with panels for:
   - LLM call latency (p50, p95, p99)
   - Token consumption over time
   - Error rates by component
   - Parallelization factor per round

## Troubleshooting

### High Memory Usage

If the observability layer consumes too much memory:

```python
# Clean up old metrics
await metrics.cleanup_old_metrics()

# Clear trace collector
await TraceCollector.clear()

# Reduce retention
metrics = MetricsCollector(retention_hours=6)  # Default is 24
```

### Missing Traces

If traces aren't appearing:

1. Check that trace_id_var is set:
   ```python
   trace_id = trace_id_var.get()
   assert trace_id is not None
   ```

2. Verify TraceContext is used as async context manager:
   ```python
   async with TraceContext("operation"):  # Correct
       ...

   # NOT: TraceContext("operation")  # Wrong
   ```

### Metrics Not Aggregating

If metrics show zero counts:

1. Check label matching is exact:
   ```python
   await metrics.record("latency", 1.0, agent="developer")

   # Must match exactly
   stats = await metrics.get_stats("latency", agent="developer")  # Works
   stats = await metrics.get_stats("latency", agent="dev")  # Returns empty
   ```

## Examples

See [examples/observability_example.py](../examples/observability_example.py) for a comprehensive demonstration.

## API Reference

For detailed API documentation, see the generated API reference (coming soon) or refer to the module docstrings:

- [observability/tracing.py](../src/dytopo/observability/tracing.py)
- [observability/metrics.py](../src/dytopo/observability/metrics.py)
- [observability/profiling.py](../src/dytopo/observability/profiling.py)
- [observability/failures.py](../src/dytopo/observability/failures.py)

## Performance Tips

1. **Use trace contexts selectively**: Only wrap significant operations
2. **Batch metric exports**: Export periodically, not after every operation
3. **Clean up regularly**: Run cleanup tasks to prevent unbounded growth
4. **Use labels wisely**: Too many unique label combinations can cause memory issues

## Future Enhancements

- [ ] Real-time metrics dashboard (Grafana integration)
- [ ] Automatic anomaly detection
- [ ] Performance regression testing
- [ ] Cost tracking (LLM API costs)
- [ ] Multi-swarm comparison
