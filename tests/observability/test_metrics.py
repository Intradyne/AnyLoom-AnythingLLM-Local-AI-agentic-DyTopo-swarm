"""
Tests for metrics collection system.
"""

import asyncio
import pytest
from pathlib import Path

from dytopo.observability.metrics import MetricsCollector


@pytest.mark.asyncio
async def test_metrics_basic():
    """Test basic metric recording."""
    collector = MetricsCollector()

    await collector.record("test_metric", 1.5, label="test")
    await collector.record("test_metric", 2.5, label="test")
    await collector.record("test_metric", 3.5, label="test")

    stats = await collector.get_stats("test_metric", label="test")

    assert stats["count"] == 3
    assert stats["sum"] == 7.5
    assert stats["mean"] == 2.5
    assert stats["min"] == 1.5
    assert stats["max"] == 3.5


@pytest.mark.asyncio
async def test_metrics_percentiles():
    """Test percentile calculations."""
    collector = MetricsCollector()

    # Record 100 values
    for i in range(100):
        await collector.record("latency", float(i), service="api")

    stats = await collector.get_stats("latency", service="api")

    assert stats["count"] == 100
    assert stats["p50"] == pytest.approx(49, abs=1)
    assert stats["p95"] == pytest.approx(94, abs=1)
    assert stats["p99"] == pytest.approx(98, abs=1)


@pytest.mark.asyncio
async def test_metrics_labels():
    """Test label-based filtering."""
    collector = MetricsCollector()

    await collector.record("latency", 1.0, service="api", status="success")
    await collector.record("latency", 2.0, service="api", status="error")
    await collector.record("latency", 3.0, service="db", status="success")

    # Filter by service
    api_stats = await collector.get_stats("latency", service="api")
    assert api_stats["count"] == 2

    # Filter by service and status
    api_success = await collector.get_stats(
        "latency",
        service="api",
        status="success"
    )
    assert api_success["count"] == 1
    assert api_success["mean"] == 1.0


@pytest.mark.asyncio
async def test_metrics_export_prometheus():
    """Test Prometheus format export."""
    collector = MetricsCollector()

    await collector.record("http_requests_total", 100, method="GET", status="200")
    await collector.record("http_requests_total", 50, method="POST", status="200")

    output_path = Path("test_metrics.prom")

    try:
        await collector.export_prometheus(output_path)

        assert output_path.exists()

        content = output_path.read_text()

        assert "# TYPE http_requests_total gauge" in content
        assert 'method="GET"' in content
        assert 'method="POST"' in content

    finally:
        if output_path.exists():
            output_path.unlink()


@pytest.mark.asyncio
async def test_metrics_cleanup():
    """Test cleanup of old metrics."""
    collector = MetricsCollector(retention_hours=0.0000001)  # ~0.36ms

    await collector.record("old_metric", 1.0)
    await asyncio.sleep(0.01)  # Wait 10ms — well past retention
    await collector.record("new_metric", 2.0)

    removed = await collector.cleanup_old_metrics()

    assert removed == 1

    all_metrics = await collector.get_all_metrics()
    assert len(all_metrics) == 1
    assert all_metrics[0].name == "new_metric"
