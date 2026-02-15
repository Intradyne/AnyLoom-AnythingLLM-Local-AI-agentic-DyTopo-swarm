"""
Tests for distributed tracing infrastructure.
"""

import asyncio
import pytest
from pathlib import Path

from dytopo.observability.tracing import (
    TraceContext,
    TraceCollector,
    trace_id_var,
    span_stack_var,
)


@pytest.mark.asyncio
async def test_trace_context_basic():
    """Test basic trace context creation and span tracking."""
    # Clear any existing state
    await TraceCollector.clear()

    async with TraceContext("test_operation", foo="bar"):
        trace_id = trace_id_var.get()
        assert trace_id is not None
        assert len(span_stack_var.get()) == 1

    # Span should be popped from stack
    assert len(span_stack_var.get()) == 0

    # Span should be recorded
    spans = await TraceCollector.get_trace(trace_id)
    assert len(spans) == 1
    assert spans[0].operation == "test_operation"
    assert spans[0].status == "success"
    assert spans[0].metadata["foo"] == "bar"


@pytest.mark.asyncio
async def test_trace_context_nested():
    """Test nested trace contexts create parent-child relationships."""
    await TraceCollector.clear()

    async with TraceContext("parent"):
        parent_span_id = span_stack_var.get()[-1]

        async with TraceContext("child"):
            child_span_id = span_stack_var.get()[-1]
            assert len(span_stack_var.get()) == 2

        # Child should be popped
        assert len(span_stack_var.get()) == 1

    trace_id = trace_id_var.get()
    spans = await TraceCollector.get_trace(trace_id)

    assert len(spans) == 2

    # Find parent and child spans
    parent = next(s for s in spans if s.operation == "parent")
    child = next(s for s in spans if s.operation == "child")

    # Verify parent-child relationship
    assert child.parent_id == parent.span_id


@pytest.mark.asyncio
async def test_trace_context_error_handling():
    """Test that errors are captured in span metadata."""
    await TraceCollector.clear()

    try:
        async with TraceContext("failing_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass  # Expected

    trace_id = trace_id_var.get()
    spans = await TraceCollector.get_trace(trace_id)

    assert len(spans) == 1
    assert spans[0].status == "error"
    assert spans[0].metadata["error_type"] == "ValueError"
    assert "Test error" in spans[0].metadata["error_message"]


@pytest.mark.asyncio
async def test_trace_export():
    """Test exporting trace to JSON."""
    await TraceCollector.clear()

    async with TraceContext("operation1"):
        await asyncio.sleep(0.01)

    async with TraceContext("operation2"):
        await asyncio.sleep(0.01)

    trace_id = trace_id_var.get()
    output_path = Path("test_trace.json")

    try:
        await TraceCollector.export_trace(trace_id, output_path)

        assert output_path.exists()

        import json
        data = json.loads(output_path.read_text())

        assert data["trace_id"] == trace_id
        assert data["total_spans"] == 2
        assert len(data["spans"]) == 2

    finally:
        if output_path.exists():
            output_path.unlink()


@pytest.mark.asyncio
async def test_trace_analysis():
    """Test trace analysis for performance insights."""
    await TraceCollector.clear()

    async with TraceContext("fast_op"):
        await asyncio.sleep(0.01)

    async with TraceContext("fast_op"):
        await asyncio.sleep(0.01)

    async with TraceContext("slow_op"):
        await asyncio.sleep(0.05)

    trace_id = trace_id_var.get()
    analysis = await TraceCollector.analyze_trace(trace_id)

    assert analysis["total_spans"] == 3
    assert "fast_op" in analysis["operation_stats"]
    assert "slow_op" in analysis["operation_stats"]

    fast_stats = analysis["operation_stats"]["fast_op"]
    assert fast_stats["count"] == 2

    slow_stats = analysis["operation_stats"]["slow_op"]
    assert slow_stats["count"] == 1
    assert slow_stats["avg_duration"] > fast_stats["avg_duration"]
