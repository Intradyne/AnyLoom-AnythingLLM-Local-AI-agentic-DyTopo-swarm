"""
DyTopo Observability Layer
==========================

Comprehensive observability infrastructure for distributed multi-agent swarms:
- Distributed tracing with trace propagation
- Enhanced metrics collection with percentiles
- Performance profiling and bottleneck analysis
- Failure analysis and pattern detection

Usage:
    from dytopo.observability import TraceContext, metrics, profiler

    async with TraceContext("agent_execution", agent_id="developer"):
        result = await execute_agent()

    await metrics.record("llm_latency_seconds", 1.5, agent="developer")
"""

from dytopo.observability.tracing import (
    TraceContext,
    TraceCollector,
    Span,
    trace_id_var,
    span_stack_var,
)

from dytopo.observability.metrics import (
    MetricsCollector,
    MetricPoint,
    metrics,
)

from dytopo.observability.profiling import (
    PerformanceProfiler,
    BottleneckAnalyzer,
    profiler,
)

from dytopo.observability.failures import (
    FailureAnalyzer,
    FailureRecord,
    failure_analyzer,
)

__all__ = [
    # Tracing
    "TraceContext",
    "TraceCollector",
    "Span",
    "trace_id_var",
    "span_stack_var",
    # Metrics
    "MetricsCollector",
    "MetricPoint",
    "metrics",
    # Profiling
    "PerformanceProfiler",
    "BottleneckAnalyzer",
    "profiler",
    # Failures
    "FailureAnalyzer",
    "FailureRecord",
    "failure_analyzer",
]
