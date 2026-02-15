"""
Performance Profiling and Bottleneck Analysis
===============================================

Tools for profiling swarm execution, identifying bottlenecks, and
generating optimization recommendations.

Features:
- CPU profiling with cProfile integration
- Async operation timing analysis
- Bottleneck detection with actionable recommendations
- Performance score calculation
- Parallelization opportunity analysis

Usage:
    from dytopo.observability import profiler

    # Profile a swarm execution
    report = await profiler.profile_swarm(orchestrator, task="prove theorem")

    # Analyze bottlenecks
    analysis = await BottleneckAnalyzer.analyze(trace_id)
    for rec in analysis["recommendations"]:
        print(f"{rec['priority']}: {rec['message']}")
"""

from __future__ import annotations

import cProfile
import logging
import pstats
import statistics
import time
from io import StringIO
from typing import Any, Dict, List

from dytopo.observability.tracing import TraceCollector, trace_id_var

logger = logging.getLogger("dytopo.observability.profiling")


class PerformanceProfiler:
    """
    Profile swarm execution to identify bottlenecks.

    Combines CPU profiling (cProfile) with async trace analysis
    to provide comprehensive performance insights.

    Usage:
        profiler = PerformanceProfiler()
        report = await profiler.profile_swarm(orchestrator, task)
        print(report["trace_analysis"]["total_duration"])
    """

    def __init__(self):
        """Initialize profiler."""
        self.profiler: Optional[cProfile.Profile] = None
        self.async_timings: Dict[str, List[float]] = {}

    async def profile_swarm(
        self,
        orchestrator_run_coro,
        task: str,
        enable_cpu_profile: bool = True,
    ) -> Dict[str, Any]:
        """
        Profile a full swarm execution.

        Args:
            orchestrator_run_coro: Coroutine for orchestrator.run(task)
            task: The task to execute
            enable_cpu_profile: Whether to run CPU profiling (adds overhead)

        Returns:
            Dict with comprehensive performance report:
            - wall_time: Total execution time
            - cpu_profile: Top 20 functions by cumulative time
            - trace_analysis: Aggregated span statistics
            - bottlenecks: Identified performance issues
            - recommendations: Optimization suggestions
        """
        # Start CPU profiler if enabled
        if enable_cpu_profile:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

        # Run swarm
        start_time = time.time()

        try:
            result = await orchestrator_run_coro
        except Exception as e:
            logger.error(f"Swarm execution failed during profiling: {e}")
            raise
        finally:
            wall_time = time.time() - start_time

            # Stop CPU profiler
            if enable_cpu_profile and self.profiler:
                self.profiler.disable()

        # Analyze CPU profile
        cpu_profile = ""
        if enable_cpu_profile and self.profiler:
            s = StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats("cumulative")
            ps.print_stats(20)  # Top 20 functions
            cpu_profile = s.getvalue()

        # Get trace analysis
        trace_id = trace_id_var.get()
        trace_analysis = {}
        bottleneck_analysis = {}

        if trace_id:
            trace_analysis = await TraceCollector.analyze_trace(trace_id)
            bottleneck_analysis = await BottleneckAnalyzer.analyze(trace_id)

        return {
            "wall_time": wall_time,
            "cpu_profile": cpu_profile,
            "trace_analysis": trace_analysis,
            "bottleneck_analysis": bottleneck_analysis,
            "task": task,
        }

    def record_async_timing(self, operation: str, duration: float):
        """
        Record timing for an async operation.

        Args:
            operation: Name of the operation
            duration: Duration in seconds
        """
        if operation not in self.async_timings:
            self.async_timings[operation] = []
        self.async_timings[operation].append(duration)

    def _percentile(self, values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class BottleneckAnalyzer:
    """
    Analyze traces and metrics to identify bottlenecks.

    Provides actionable recommendations for performance optimization.
    """

    @staticmethod
    async def analyze(trace_id: str) -> Dict[str, Any]:
        """
        Analyze trace for performance issues.

        Detects:
        - Slow operations (>10s)
        - Low parallelization (sequential bottlenecks)
        - High-frequency operations (caching opportunities)
        - Unbalanced execution (some agents much slower)

        Args:
            trace_id: Trace ID to analyze

        Returns:
            Dict with:
            - total_duration: Total trace duration
            - parallelization_factor: Ratio of total work to wall time
            - slowest_operations: Top 10 slowest operations
            - long_poles: Operations blocking parallel progress
            - recommendations: List of optimization suggestions
            - performance_score: 0-100 score
        """
        trace = await TraceCollector.get_trace(trace_id)

        if not trace:
            return {"error": f"No trace found for {trace_id}"}

        # Find slowest operations
        slow_ops = sorted(
            [s for s in trace if s.duration],
            key=lambda s: s.duration,
            reverse=True,
        )[:10]

        # Calculate parallelization opportunity
        total_sequential = sum(s.duration for s in trace if s.duration)
        total_wall = (
            max(s.end_time for s in trace if s.end_time) -
            min(s.start_time for s in trace)
        )
        parallelization_factor = total_sequential / total_wall if total_wall > 0 else 1.0

        # Identify long-pole operations (blocking parallel progress)
        long_poles = []
        for span in trace:
            if span.duration and span.duration > 10.0:  # >10s is significant
                long_poles.append({
                    "operation": span.operation,
                    "duration": span.duration,
                    "metadata": span.metadata,
                })

        # Generate recommendations
        recommendations = []

        if parallelization_factor < 2.0:
            recommendations.append({
                "priority": "high",
                "category": "parallelization",
                "message": (
                    f"Low parallelization factor ({parallelization_factor:.1f}x). "
                    "Consider increasing concurrent execution or reducing sequential bottlenecks."
                ),
            })

        if long_poles:
            recommendations.append({
                "priority": "high",
                "category": "long_poles",
                "message": (
                    f"Found {len(long_poles)} operations >10s. "
                    "These block parallel progress and should be optimized."
                ),
                "details": long_poles,
            })

        # Check for repeated operations (caching opportunities)
        op_counts: Dict[str, int] = {}
        for span in trace:
            op_counts[span.operation] = op_counts.get(span.operation, 0) + 1

        high_frequency = {op: count for op, count in op_counts.items() if count > 20}
        if high_frequency:
            recommendations.append({
                "priority": "medium",
                "category": "caching",
                "message": (
                    f"High-frequency operations detected: {high_frequency}. "
                    "Consider caching or batching these operations."
                ),
            })

        # Check for errors
        error_count = sum(1 for s in trace if s.status == "error")
        if error_count > 0:
            error_rate = error_count / len(trace)
            recommendations.append({
                "priority": "high" if error_rate > 0.1 else "medium",
                "category": "errors",
                "message": (
                    f"{error_count} errors detected ({error_rate:.1%} error rate). "
                    "Investigate and fix failing operations."
                ),
            })

        # Calculate performance score
        # Based on: parallelization (40%), no long poles (30%), low errors (30%)
        score = 0
        score += min(40, int(parallelization_factor * 20))  # Max 40 points
        score += 30 if not long_poles else max(0, 30 - len(long_poles) * 5)  # Max 30 points
        score += max(0, 30 - error_count * 5)  # Max 30 points

        return {
            "trace_id": trace_id,
            "total_duration": total_wall,
            "total_spans": len(trace),
            "parallelization_factor": parallelization_factor,
            "slowest_operations": [
                {"operation": s.operation, "duration": s.duration}
                for s in slow_ops
            ],
            "long_poles": long_poles,
            "error_count": error_count,
            "recommendations": recommendations,
            "performance_score": min(100, score),
        }

    @staticmethod
    def format_report(analysis: Dict[str, Any]) -> str:
        """
        Format bottleneck analysis as human-readable report.

        Args:
            analysis: Output from BottleneckAnalyzer.analyze()

        Returns:
            Formatted text report
        """
        lines = [
            "=" * 70,
            "PERFORMANCE ANALYSIS REPORT",
            "=" * 70,
            "",
            f"Trace ID: {analysis.get('trace_id', 'unknown')}",
            f"Total Duration: {analysis.get('total_duration', 0):.2f}s",
            f"Total Spans: {analysis.get('total_spans', 0)}",
            f"Parallelization Factor: {analysis.get('parallelization_factor', 0):.1f}x",
            f"Performance Score: {analysis.get('performance_score', 0)}/100",
            "",
            "SLOWEST OPERATIONS:",
            "-" * 70,
        ]

        for i, op in enumerate(analysis.get("slowest_operations", [])[:5], 1):
            lines.append(
                f"{i}. {op['operation']}: {op['duration']:.3f}s"
            )

        lines.extend(["", "RECOMMENDATIONS:", "-" * 70])

        for rec in analysis.get("recommendations", []):
            priority_mark = {
                "high": "[!!!]",
                "medium": "[!!]",
                "low": "[!]",
            }.get(rec.get("priority", "low"), "[!]")

            lines.append(f"{priority_mark} {rec.get('category', 'general').upper()}")
            lines.append(f"    {rec.get('message', '')}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


# Global singleton profiler
profiler = PerformanceProfiler()
