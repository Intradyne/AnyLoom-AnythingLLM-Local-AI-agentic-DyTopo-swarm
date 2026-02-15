"""
Enhanced Metrics Collection for DyTopo Swarms
==============================================

Comprehensive metrics collection with statistical aggregation,
percentile calculations, and Prometheus-compatible export.

Metric Categories:
1. Latency - Duration of operations (LLM calls, routing, agent execution)
2. Throughput - Requests/tokens per second
3. Errors - Failure counts and rates by type
4. Resources - Token usage, memory consumption

Features:
- Automatic percentile calculation (p50, p95, p99)
- Label-based filtering for multi-dimensional analysis
- Prometheus text format export
- Zero-overhead recording via async

Usage:
    from dytopo.observability import metrics

    # Record a latency metric
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
    print(f"p95: {stats['p95']:.3f}s")

    # Export to Prometheus format
    await metrics.export_prometheus(Path("metrics.prom"))
"""

from __future__ import annotations

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("dytopo.observability.metrics")


@dataclass
class MetricPoint:
    """
    A single metric observation.

    Attributes:
        name: Metric name (e.g., "llm_call_duration_seconds")
        value: Numeric value
        timestamp: Unix timestamp when metric was recorded
        labels: Key-value pairs for filtering (e.g., {"agent": "developer"})
    """

    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates metrics for analysis.

    Thread-safe metric collection with support for:
    - Label-based filtering
    - Statistical aggregation (min, max, mean, median, percentiles)
    - Prometheus text format export
    - Automatic metric retention (configurable)

    Usage:
        collector = MetricsCollector(retention_hours=24)

        await collector.record("latency", 1.5, service="api", status="ok")
        stats = await collector.get_stats("latency", service="api")
        print(f"Average: {stats['mean']:.3f}")
    """

    def __init__(self, retention_hours: float = 24.0):
        """
        Initialize metrics collector.

        Args:
            retention_hours: How long to keep metrics before auto-cleanup
        """
        self._metrics: List[MetricPoint] = []
        self._lock = asyncio.Lock()
        self._retention_seconds = retention_hours * 3600

    async def record(
        self,
        name: str,
        value: float,
        **labels: str
    ):
        """
        Record a metric point.

        Args:
            name: Metric name (lowercase with underscores)
            value: Numeric value to record
            **labels: Key-value pairs for filtering
        """
        async with self._lock:
            self._metrics.append(
                MetricPoint(
                    name=name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels,
                )
            )

    async def get_stats(
        self,
        name: str,
        **filter_labels: str
    ) -> Dict[str, Any]:
        """
        Get statistics for a metric.

        Args:
            name: Metric name to query
            **filter_labels: Filter by label values (e.g., agent="developer")

        Returns:
            Dict with statistical summaries:
            - count: Number of observations
            - sum: Total sum
            - min: Minimum value
            - max: Maximum value
            - mean: Average value
            - median: Median value
            - p50, p95, p99: Percentiles
        """
        async with self._lock:
            # Filter metrics
            matching = [
                m for m in self._metrics
                if m.name == name and
                all(m.labels.get(k) == v for k, v in filter_labels.items())
            ]

            if not matching:
                return {}

            values = [m.value for m in matching]

            return {
                "count": len(values),
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "p50": self._percentile(values, 0.50),
                "p95": self._percentile(values, 0.95),
                "p99": self._percentile(values, 0.99),
            }

    async def get_all_metrics(self) -> List[MetricPoint]:
        """
        Get all recorded metrics.

        Returns:
            List of all metric points
        """
        async with self._lock:
            return list(self._metrics)

    async def cleanup_old_metrics(self):
        """
        Remove metrics older than retention period.

        Returns:
            Number of metrics removed
        """
        cutoff_time = time.time() - self._retention_seconds

        async with self._lock:
            original_count = len(self._metrics)
            self._metrics = [
                m for m in self._metrics
                if m.timestamp >= cutoff_time
            ]
            removed = original_count - len(self._metrics)

        if removed > 0:
            logger.info(f"Cleaned up {removed} old metrics")

        return removed

    async def export_prometheus(self, output_path: Path):
        """
        Export metrics in Prometheus text format.

        Format:
            # TYPE metric_name gauge
            metric_name{label1="value1",label2="value2"} 123.45 1234567890000

        Args:
            output_path: Where to write the metrics file
        """
        async with self._lock:
            lines = []

            # Group by metric name
            by_name: Dict[str, List[MetricPoint]] = {}
            for m in self._metrics:
                if m.name not in by_name:
                    by_name[m.name] = []
                by_name[m.name].append(m)

            # Format each metric group
            for name, points in sorted(by_name.items()):
                lines.append(f"# TYPE {name} gauge")

                for p in points:
                    # Format labels
                    if p.labels:
                        label_str = ",".join(
                            f'{k}="{v}"' for k, v in sorted(p.labels.items())
                        )
                        metric_line = f"{name}{{{label_str}}} {p.value} {int(p.timestamp * 1000)}"
                    else:
                        metric_line = f"{name} {p.value} {int(p.timestamp * 1000)}"

                    lines.append(metric_line)

                lines.append("")  # Blank line between metric groups

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info(
            f"Exported {len(self._metrics)} metrics to {output_path} "
            f"({len(by_name)} unique metric names)"
        )

    async def export_json(self, output_path: Path):
        """
        Export metrics as JSON.

        Args:
            output_path: Where to write the JSON file
        """
        import json

        async with self._lock:
            data = {
                "total_metrics": len(self._metrics),
                "retention_hours": self._retention_seconds / 3600,
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp,
                        "labels": m.labels,
                    }
                    for m in self._metrics
                ],
            }

        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON
        output_path.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

        logger.info(f"Exported {len(self._metrics)} metrics to {output_path}")

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all collected metrics.

        Returns:
            Dict with overall statistics
        """
        async with self._lock:
            if not self._metrics:
                return {"total_metrics": 0}

            # Group by name
            by_name: Dict[str, List[float]] = {}
            for m in self._metrics:
                if m.name not in by_name:
                    by_name[m.name] = []
                by_name[m.name].append(m.value)

            # Calculate stats per metric
            summary = {}
            for name, values in by_name.items():
                summary[name] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                }

            return {
                "total_metrics": len(self._metrics),
                "unique_metric_names": len(by_name),
                "oldest_metric": min(m.timestamp for m in self._metrics),
                "newest_metric": max(m.timestamp for m in self._metrics),
                "metrics": summary,
            }

    async def clear(self):
        """Clear all metrics (useful for testing)."""
        async with self._lock:
            count = len(self._metrics)
            self._metrics.clear()

        logger.info(f"Cleared {count} metrics")

    def _percentile(self, values: List[float], p: float) -> float:
        """
        Calculate percentile.

        Args:
            values: List of numeric values
            p: Percentile (0.0 to 1.0)

        Returns:
            Value at the given percentile
        """
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        # Clamp to valid range
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


# Global singleton metrics collector
metrics = MetricsCollector(retention_hours=24.0)
