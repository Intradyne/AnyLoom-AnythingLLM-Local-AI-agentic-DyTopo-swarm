"""
Distributed Tracing for DyTopo Swarms
======================================

Provides distributed tracing capabilities for tracking execution across
async multi-agent swarms. Implements OpenTelemetry-style trace propagation
using Python contextvars for seamless async context management.

Key Features:
- Automatic trace ID generation and propagation
- Hierarchical span tracking (parent-child relationships)
- JSON export for external analysis (Jaeger, Zipkin compatible)
- Zero-config instrumentation via context managers
- Per-span metadata and error capture

Usage:
    async with TraceContext("swarm_execution", task="prove theorem"):
        for round in range(T_max):
            async with TraceContext("round", round_num=round):
                async with TraceContext("agent_call", agent="developer"):
                    result = await call_agent()

    # Export trace for analysis
    trace_id = trace_id_var.get()
    await TraceCollector.export_trace(trace_id, Path("traces/trace.json"))
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dytopo.observability.tracing")

# Global context variables for trace propagation across async calls
trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
span_stack_var: ContextVar[List[str]] = ContextVar("span_stack", default=[])


@dataclass
class Span:
    """
    A single unit of work in a distributed trace.

    Represents one operation (e.g., "agent_execution", "llm_call", "routing")
    with timing, metadata, and error tracking.

    Attributes:
        span_id: Unique identifier for this span (hex string)
        parent_id: ID of parent span (None for root span)
        trace_id: ID of the overall trace this span belongs to
        operation: Name of the operation (e.g., "manager_call")
        start_time: Unix timestamp when span started
        end_time: Unix timestamp when span finished (None if still running)
        status: "in_progress" | "success" | "error"
        metadata: Additional key-value pairs for context
    """

    span_id: str
    parent_id: Optional[str]
    trace_id: str
    operation: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "in_progress"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate span_id if not provided."""
        if self.span_id is None:
            self.span_id = uuid.uuid4().hex[:16]

    def finish(self, status: str = "success", **metadata):
        """
        Mark span as complete and record final status.

        Args:
            status: Final status ("success" or "error")
            **metadata: Additional metadata to attach to span
        """
        self.end_time = time.time()
        self.status = status
        self.metadata.update(metadata)

    @property
    def duration(self) -> Optional[float]:
        """
        Calculate span duration in seconds.

        Returns:
            Duration in seconds, or None if span hasn't finished
        """
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize span to dictionary for JSON export.

        Returns:
            Dict with all span fields
        """
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "operation": self.operation,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "status": self.status,
            "metadata": self.metadata,
        }


class TraceContext:
    """
    Async context manager for distributed tracing.

    Automatically creates a span on entry, tracks timing, captures errors,
    and exports to TraceCollector on exit. Handles parent-child span
    relationships via contextvars.

    Usage:
        async with TraceContext("agent_execution", agent_id="developer"):
            result = await execute_agent()

        # Nested spans automatically get parent relationship:
        async with TraceContext("swarm_execution"):
            async with TraceContext("round", round_num=1):
                async with TraceContext("agent_call"):
                    ...  # This span has "round" as parent
    """

    def __init__(self, operation: str, **metadata):
        """
        Initialize trace context.

        Args:
            operation: Name of the operation being traced
            **metadata: Additional context to attach to the span
        """
        self.operation = operation
        self.metadata = metadata
        self.span: Optional[Span] = None

    async def __aenter__(self) -> Span:
        """
        Enter context: create span and push to stack.

        Returns:
            The created Span object
        """
        # Get or create trace ID
        trace_id = trace_id_var.get()
        if trace_id is None:
            trace_id = uuid.uuid4().hex
            trace_id_var.set(trace_id)

        # Get parent span from stack
        span_stack = span_stack_var.get()
        parent_id = span_stack[-1] if span_stack else None

        # Create new span
        self.span = Span(
            span_id=uuid.uuid4().hex[:16],
            parent_id=parent_id,
            trace_id=trace_id,
            operation=self.operation,
            start_time=time.time(),
            metadata=self.metadata,
        )

        # Push span ID to stack
        new_stack = span_stack + [self.span.span_id]
        span_stack_var.set(new_stack)

        # Log span start
        logger.debug(
            f"[TRACE] Started {self.operation} "
            f"(span={self.span.span_id}, trace={trace_id})"
        )

        return self.span

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit context: finish span, pop from stack, record to collector.

        Args:
            exc_type: Exception type if error occurred
            exc_val: Exception value if error occurred
            exc_tb: Exception traceback if error occurred

        Returns:
            False (don't suppress exceptions)
        """
        # Pop from stack
        span_stack = span_stack_var.get()
        if span_stack:
            span_stack_var.set(span_stack[:-1])

        # Finish span with error info if applicable
        if exc_type is not None:
            self.span.finish(
                status="error",
                error_type=exc_type.__name__,
                error_message=str(exc_val),
            )
        else:
            self.span.finish(status="success")

        # Log span completion
        logger.debug(
            f"[TRACE] Finished {self.operation} "
            f"({self.span.duration:.3f}s, {self.span.status})"
        )

        # Record to trace collector
        await TraceCollector.record(self.span)

        return False  # Don't suppress exceptions


class TraceCollector:
    """
    Singleton collector for trace spans.

    Stores completed spans in memory and provides export/analysis capabilities.
    Thread-safe using asyncio locks.

    Usage:
        # Automatic recording via TraceContext
        async with TraceContext("operation"):
            ...

        # Manual querying
        trace_id = trace_id_var.get()
        spans = await TraceCollector.get_trace(trace_id)

        # Export to JSON
        await TraceCollector.export_trace(trace_id, Path("trace.json"))

        # Analyze for bottlenecks
        analysis = await TraceCollector.analyze_trace(trace_id)
    """

    _spans: Dict[str, List[Span]] = {}  # trace_id -> [spans]
    _lock = asyncio.Lock()

    @classmethod
    async def record(cls, span: Span):
        """
        Record a completed span.

        Args:
            span: The span to record
        """
        async with cls._lock:
            if span.trace_id not in cls._spans:
                cls._spans[span.trace_id] = []
            cls._spans[span.trace_id].append(span)

    @classmethod
    async def get_trace(cls, trace_id: str) -> List[Span]:
        """
        Get all spans for a specific trace.

        Args:
            trace_id: The trace ID to query

        Returns:
            List of spans in chronological order
        """
        async with cls._lock:
            return list(cls._spans.get(trace_id, []))

    @classmethod
    async def get_all_traces(cls) -> Dict[str, List[Span]]:
        """
        Get all collected traces.

        Returns:
            Dict mapping trace_id -> list of spans
        """
        async with cls._lock:
            return dict(cls._spans)

    @classmethod
    async def export_trace(cls, trace_id: str, output_path: Path):
        """
        Export a trace to JSON file.

        Format is compatible with Jaeger/Zipkin visualization tools.

        Args:
            trace_id: The trace ID to export
            output_path: Where to write the JSON file
        """
        spans = await cls.get_trace(trace_id)

        trace_data = {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "start_time": min(s.start_time for s in spans) if spans else 0,
            "end_time": max(s.end_time for s in spans if s.end_time) if spans else 0,
            "total_duration": (
                max(s.end_time for s in spans if s.end_time) -
                min(s.start_time for s in spans)
            ) if spans else 0,
            "spans": [s.to_dict() for s in spans],
        }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON with pretty formatting
        output_path.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")

        logger.info(f"Exported trace {trace_id} to {output_path} ({len(spans)} spans)")

    @classmethod
    async def analyze_trace(cls, trace_id: str) -> Dict[str, Any]:
        """
        Analyze a trace for performance insights.

        Aggregates span data by operation type to identify:
        - Total duration per operation
        - Average duration per operation
        - Error rates per operation
        - Overall trace statistics

        Args:
            trace_id: The trace ID to analyze

        Returns:
            Dict with analysis results
        """
        spans = await cls.get_trace(trace_id)

        if not spans:
            return {"error": f"No spans found for trace {trace_id}"}

        # Aggregate by operation
        operation_stats: Dict[str, Dict[str, Any]] = {}
        for span in spans:
            op = span.operation
            if op not in operation_stats:
                operation_stats[op] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "errors": 0,
                    "min_duration": float('inf'),
                    "max_duration": 0.0,
                }

            stats = operation_stats[op]
            stats["count"] += 1

            if span.duration is not None:
                stats["total_duration"] += span.duration
                stats["min_duration"] = min(stats["min_duration"], span.duration)
                stats["max_duration"] = max(stats["max_duration"], span.duration)

            if span.status == "error":
                stats["errors"] += 1

        # Calculate averages and error rates
        for op, stats in operation_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["error_rate"] = stats["errors"] / stats["count"]

            # Clean up infinity
            if stats["min_duration"] == float('inf'):
                stats["min_duration"] = 0.0

        # Overall trace statistics
        total_duration = (
            max(s.end_time for s in spans if s.end_time) -
            min(s.start_time for s in spans)
        ) if spans else 0

        total_errors = sum(1 for s in spans if s.status == "error")

        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "total_duration": total_duration,
            "total_errors": total_errors,
            "error_rate": total_errors / len(spans) if spans else 0,
            "operation_stats": operation_stats,
        }

    @classmethod
    async def clear(cls):
        """Clear all collected traces (useful for testing)."""
        async with cls._lock:
            cls._spans.clear()
