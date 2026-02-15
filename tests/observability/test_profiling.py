"""
Tests for performance profiling and bottleneck analysis.
"""

import asyncio
import pytest

from dytopo.observability.profiling import BottleneckAnalyzer
from dytopo.observability.tracing import TraceContext, TraceCollector, trace_id_var


@pytest.mark.asyncio
async def test_bottleneck_analyzer_basic():
    """Test basic bottleneck analysis."""
    await TraceCollector.clear()

    # Create a trace with various operations
    async with TraceContext("fast_operation"):
        await asyncio.sleep(0.01)

    async with TraceContext("slow_operation"):
        await asyncio.sleep(0.1)

    async with TraceContext("another_fast"):
        await asyncio.sleep(0.01)

    trace_id = trace_id_var.get()
    analysis = await BottleneckAnalyzer.analyze(trace_id)

    assert analysis["total_spans"] == 3
    assert analysis["total_duration"] > 0.1

    # Check slowest operations
    slowest = analysis["slowest_operations"]
    assert len(slowest) > 0
    assert slowest[0]["operation"] == "slow_operation"


@pytest.mark.asyncio
async def test_bottleneck_analyzer_recommendations():
    """Test that bottleneck analyzer generates recommendations."""
    await TraceCollector.clear()

    # Create a trace with low parallelization (sequential operations)
    async with TraceContext("seq1"):
        await asyncio.sleep(0.05)

    async with TraceContext("seq2"):
        await asyncio.sleep(0.05)

    async with TraceContext("seq3"):
        await asyncio.sleep(0.05)

    trace_id = trace_id_var.get()
    analysis = await BottleneckAnalyzer.analyze(trace_id)

    # Should detect low parallelization
    assert analysis["parallelization_factor"] < 2.0

    # Should have recommendations
    recommendations = analysis["recommendations"]
    assert len(recommendations) > 0

    # Check for parallelization recommendation
    parallel_rec = next(
        (r for r in recommendations if r["category"] == "parallelization"),
        None
    )
    assert parallel_rec is not None
    assert parallel_rec["priority"] == "high"


@pytest.mark.asyncio
async def test_bottleneck_analyzer_error_detection():
    """Test detection of errors in traces."""
    await TraceCollector.clear()

    async with TraceContext("good_operation"):
        pass

    try:
        async with TraceContext("bad_operation"):
            raise ValueError("Test error")
    except ValueError:
        pass

    trace_id = trace_id_var.get()
    analysis = await BottleneckAnalyzer.analyze(trace_id)

    assert analysis["error_count"] == 1

    # Should have error recommendation
    recommendations = analysis["recommendations"]
    error_recs = [r for r in recommendations if r["category"] == "errors"]
    assert len(error_recs) > 0


@pytest.mark.asyncio
async def test_format_report():
    """Test report formatting."""
    await TraceCollector.clear()

    async with TraceContext("operation"):
        await asyncio.sleep(0.01)

    trace_id = trace_id_var.get()
    analysis = await BottleneckAnalyzer.analyze(trace_id)

    report = BottleneckAnalyzer.format_report(analysis)

    assert "PERFORMANCE ANALYSIS REPORT" in report
    assert "Trace ID:" in report
    assert "Performance Score:" in report
