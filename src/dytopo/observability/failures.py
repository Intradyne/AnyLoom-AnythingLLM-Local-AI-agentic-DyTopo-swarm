"""
Enhanced Failure Analysis and Recovery
========================================

Tracks and analyzes failures for debugging and recovery strategies.

Features:
- Detailed failure event tracking
- Pattern detection (recurring failures)
- Root cause analysis
- Recovery strategy suggestions
- Failure rate monitoring

Usage:
    from dytopo.observability import failure_analyzer

    # Record a failure
    failure_id = await failure_analyzer.record_failure(
        component="orchestrator",
        operation="agent_call",
        error=TimeoutError("Agent timeout"),
        agent_id="developer",
        round=3
    )

    # Mark as recovered
    await failure_analyzer.mark_recovered(failure_id, "retry")

    # Analyze patterns
    patterns = await failure_analyzer.analyze_patterns()
    print(f"Most common error: {patterns['most_common_error']}")
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dytopo.observability.tracing import trace_id_var, span_stack_var

logger = logging.getLogger("dytopo.observability.failures")


@dataclass
class FailureRecord:
    """
    Record of a failure event.

    Attributes:
        failure_id: Unique identifier for this failure
        timestamp: When the failure occurred
        component: Which component failed (e.g., "orchestrator", "routing")
        operation: What operation failed (e.g., "agent_call", "llm_request")
        error_type: Type of exception (e.g., "TimeoutError")
        error_message: Error message or description
        trace_id: Associated trace ID (if any)
        span_id: Associated span ID (if any)
        context: Additional context (agent_id, round, etc.)
        recovered: Whether the failure was successfully recovered
        recovery_strategy: How recovery was achieved (if recovered)
    """

    failure_id: str
    timestamp: float
    component: str
    operation: str
    error_type: str
    error_message: str
    trace_id: str
    span_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    recovery_strategy: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "failure_id": self.failure_id,
            "timestamp": self.timestamp,
            "component": self.component,
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "context": self.context,
            "recovered": self.recovered,
            "recovery_strategy": self.recovery_strategy,
        }


class FailureAnalyzer:
    """
    Tracks and analyzes failures for debugging and recovery.

    Capabilities:
    - Record failure events with full context
    - Track recovery attempts
    - Detect failure patterns
    - Calculate failure rates by component/type
    - Generate recovery strategy suggestions

    Usage:
        analyzer = FailureAnalyzer()

        # Record failure
        failure_id = await analyzer.record_failure(
            "orchestrator", "llm_call", TimeoutError("Timeout"),
            agent_id="developer"
        )

        # Check patterns
        patterns = await analyzer.analyze_patterns()
    """

    def __init__(self):
        """Initialize failure analyzer."""
        self._failures: List[FailureRecord] = []
        self._lock = asyncio.Lock()

    async def record_failure(
        self,
        component: str,
        operation: str,
        error: Exception,
        **context: Any,
    ) -> str:
        """
        Record a failure event.

        Args:
            component: Component where failure occurred
            operation: Operation that failed
            error: The exception that was raised
            **context: Additional context (agent_id, round, etc.)

        Returns:
            failure_id for tracking/recovery
        """
        failure_id = uuid.uuid4().hex[:12]

        record = FailureRecord(
            failure_id=failure_id,
            timestamp=time.time(),
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            error_message=str(error),
            trace_id=trace_id_var.get() or "unknown",
            span_id=span_stack_var.get()[-1] if span_stack_var.get() else "unknown",
            context=context,
        )

        async with self._lock:
            self._failures.append(record)

        logger.error(
            f"[FAILURE] {component}.{operation} failed: {error} "
            f"(failure_id={failure_id}, trace={record.trace_id})"
        )

        return failure_id

    async def mark_recovered(self, failure_id: str, strategy: str):
        """
        Mark a failure as recovered.

        Args:
            failure_id: ID of the failure to mark
            strategy: How it was recovered (e.g., "retry", "fallback", "skip")
        """
        async with self._lock:
            for record in self._failures:
                if record.failure_id == failure_id:
                    record.recovered = True
                    record.recovery_strategy = strategy
                    logger.info(
                        f"[RECOVERY] Failure {failure_id} recovered via {strategy}"
                    )
                    break

    async def get_failure(self, failure_id: str) -> Optional[FailureRecord]:
        """
        Get a specific failure record.

        Args:
            failure_id: ID of the failure to retrieve

        Returns:
            FailureRecord or None if not found
        """
        async with self._lock:
            for record in self._failures:
                if record.failure_id == failure_id:
                    return record
        return None

    async def get_all_failures(self) -> List[FailureRecord]:
        """Get all failure records."""
        async with self._lock:
            return list(self._failures)

    async def analyze_patterns(self) -> Dict[str, Any]:
        """
        Analyze failure patterns.

        Detects:
        - Most common error types
        - Most problematic components
        - Recovery success rate
        - Temporal patterns (failure clustering)

        Returns:
            Dict with pattern analysis
        """
        async with self._lock:
            if not self._failures:
                return {"total_failures": 0}

            # Group by error type
            by_type: Dict[str, List[FailureRecord]] = {}
            for f in self._failures:
                if f.error_type not in by_type:
                    by_type[f.error_type] = []
                by_type[f.error_type].append(f)

            # Group by component
            by_component: Dict[str, List[FailureRecord]] = {}
            for f in self._failures:
                if f.component not in by_component:
                    by_component[f.component] = []
                by_component[f.component].append(f)

            # Calculate recovery rate
            recovered_count = sum(1 for f in self._failures if f.recovered)
            recovery_rate = recovered_count / len(self._failures)

            # Find most common error
            most_common_error = max(by_type.items(), key=lambda x: len(x[1]))[0]

            # Find most problematic component
            most_problematic = max(by_component.items(), key=lambda x: len(x[1]))[0]

            # Check for temporal clustering (multiple failures in short time)
            clustered_failures = self._detect_temporal_clustering()

            return {
                "total_failures": len(self._failures),
                "by_type": {t: len(fs) for t, fs in by_type.items()},
                "by_component": {c: len(fs) for c, fs in by_component.items()},
                "recovery_rate": recovery_rate,
                "recovered_count": recovered_count,
                "most_common_error": most_common_error,
                "most_problematic_component": most_problematic,
                "temporal_clusters": clustered_failures,
            }

    async def get_recovery_suggestions(
        self,
        component: str,
        error_type: str,
    ) -> List[str]:
        """
        Get recovery strategy suggestions for a specific failure type.

        Args:
            component: Component that failed
            error_type: Type of error

        Returns:
            List of suggested recovery strategies
        """
        suggestions = []

        # General suggestions by error type
        if error_type == "TimeoutError":
            suggestions.extend([
                "Increase timeout duration",
                "Retry with exponential backoff",
                "Check if service is overloaded",
                "Consider async processing",
            ])
        elif error_type == "ConnectionError":
            suggestions.extend([
                "Verify service is running",
                "Check network connectivity",
                "Retry with exponential backoff",
                "Implement circuit breaker",
            ])
        elif error_type == "JSONDecodeError":
            suggestions.extend([
                "Enable JSON repair fallback",
                "Validate LLM response format",
                "Add response schema validation",
                "Log raw response for debugging",
            ])
        elif "OutOfMemory" in error_type or "OOM" in error_type:
            suggestions.extend([
                "Reduce batch size",
                "Implement memory cleanup",
                "Add memory monitoring",
                "Split work into smaller chunks",
            ])
        else:
            suggestions.extend([
                "Add detailed error logging",
                "Implement graceful degradation",
                "Add error recovery retry logic",
            ])

        # Component-specific suggestions
        if component == "orchestrator":
            suggestions.append("Consider fallback to broadcast mode")
        elif component == "routing":
            suggestions.append("Fall back to previous routing graph")
        elif component == "agent":
            suggestions.append("Mark agent as failed and continue swarm")

        return suggestions

    def _detect_temporal_clustering(self, window_seconds: float = 60.0) -> int:
        """
        Detect if failures are clustered in time.

        Args:
            window_seconds: Time window for clustering detection

        Returns:
            Number of failure clusters detected
        """
        if len(self._failures) < 3:
            return 0

        # Sort by timestamp
        sorted_failures = sorted(self._failures, key=lambda f: f.timestamp)

        clusters = 0
        current_cluster_start = sorted_failures[0].timestamp
        cluster_count = 1

        for i in range(1, len(sorted_failures)):
            time_delta = sorted_failures[i].timestamp - sorted_failures[i - 1].timestamp

            if time_delta <= window_seconds:
                cluster_count += 1
            else:
                # Cluster ended
                if cluster_count >= 3:  # At least 3 failures to be a cluster
                    clusters += 1
                current_cluster_start = sorted_failures[i].timestamp
                cluster_count = 1

        # Check last cluster
        if cluster_count >= 3:
            clusters += 1

        return clusters

    async def export_failures_json(self, output_path) -> None:
        """
        Export all failures to JSON file.

        Args:
            output_path: Where to write the JSON file
        """
        import json
        from pathlib import Path

        async with self._lock:
            data = {
                "total_failures": len(self._failures),
                "failures": [f.to_dict() for f in self._failures],
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(
            json.dumps(data, indent=2),
            encoding="utf-8"
        )

        logger.info(f"Exported {len(self._failures)} failures to {output_path}")

    async def clear(self):
        """Clear all failure records (useful for testing)."""
        async with self._lock:
            count = len(self._failures)
            self._failures.clear()

        logger.info(f"Cleared {count} failure records")


# Global singleton failure analyzer
failure_analyzer = FailureAnalyzer()
