"""
Delegation System for DyTopo Swarms
=====================================

Manages task delegation with concurrency control, depth limits,
timeouts, and lineage tracking.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("dytopo.delegation")


class DelegationError(Exception):
    """Raised when delegation fails (depth exceeded, timeout, runner error)."""


@dataclass
class DelegationRecord:
    """Tracks a single delegation's lifecycle.

    Attributes:
        delegation_id: Unique identifier for this delegation.
        parent_agent_id: Agent that requested the delegation.
        subtask: Description of the delegated subtask.
        depth: Nesting depth (0 = top-level).
        status: Current status (pending/running/completed/failed/timed_out).
        result: Output from the delegation (if completed).
        error: Error message (if failed or timed_out).
        start_time: When the delegation started.
        end_time: When the delegation finished.
    """

    delegation_id: str
    parent_agent_id: str
    subtask: str
    depth: int
    status: str = "pending"  # pending | running | completed | failed | timed_out
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: float = 0.0
    end_time: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


class DelegationManager:
    """Manages task delegation with safety constraints.

    Controls concurrency, depth, and timeout for delegated subtasks.
    The actual work is performed by an injected `swarm_runner` callable,
    keeping delegation logic decoupled from orchestration.

    Usage:
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=300.0)

        async def my_runner(subtask: str, context: dict) -> str:
            return await run_mini_swarm(subtask, context)

        result = await manager.delegate(
            parent_agent_id="developer",
            subtask="verify the proof",
            context={"domain": "math"},
            swarm_runner=my_runner,
        )
    """

    def __init__(
        self,
        max_depth: int = 2,
        max_concurrent: int = 4,
        timeout: float = 300.0,
    ):
        self.max_depth = max_depth
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tree: Dict[str, List[str]] = {}  # parent_id -> [delegation_ids]
        self._records: Dict[str, DelegationRecord] = {}
        self._active_count: int = 0

    async def delegate(
        self,
        parent_agent_id: str,
        subtask: str,
        context: Dict[str, Any],
        swarm_runner: Callable[..., Awaitable[str]],
        depth: int = 0,
    ) -> str:
        """Delegate a subtask with concurrency and depth control.

        Args:
            parent_agent_id: ID of the agent requesting delegation.
            subtask: Description of what needs to be done.
            context: Additional context for the runner.
            swarm_runner: Async callable(subtask, context) -> result string.
            depth: Current delegation depth (0 = first level).

        Returns:
            Result string from the swarm runner.

        Raises:
            DelegationError: If depth exceeded, timeout, or runner failure.
        """
        if depth >= self.max_depth:
            raise DelegationError(
                f"Max delegation depth ({self.max_depth}) exceeded at depth {depth}"
            )

        delegation_id = f"{parent_agent_id}_sub_{uuid.uuid4().hex[:8]}"
        record = DelegationRecord(
            delegation_id=delegation_id,
            parent_agent_id=parent_agent_id,
            subtask=subtask,
            depth=depth,
            start_time=time.monotonic(),
        )
        self._records[delegation_id] = record

        # Track in tree
        if parent_agent_id not in self._tree:
            self._tree[parent_agent_id] = []
        self._tree[parent_agent_id].append(delegation_id)

        async with self._semaphore:
            self._active_count += 1
            record.status = "running"
            try:
                result = await asyncio.wait_for(
                    swarm_runner(subtask, context),
                    timeout=self.timeout,
                )
                record.status = "completed"
                record.result = result
                record.end_time = time.monotonic()
                logger.info(
                    f"Delegation {delegation_id} completed "
                    f"({record.duration:.1f}s, depth={depth})"
                )
                return result

            except asyncio.TimeoutError:
                record.status = "timed_out"
                record.error = f"Timed out after {self.timeout}s"
                record.end_time = time.monotonic()
                logger.error(f"Delegation {delegation_id} timed out")
                raise DelegationError(
                    f"Delegation {delegation_id} timed out after {self.timeout}s"
                )

            except DelegationError:
                record.status = "failed"
                record.end_time = time.monotonic()
                raise

            except Exception as e:
                record.status = "failed"
                record.error = str(e)
                record.end_time = time.monotonic()
                logger.error(f"Delegation {delegation_id} failed: {e}")
                raise DelegationError(
                    f"Delegation {delegation_id} failed: {e}"
                ) from e

            finally:
                self._active_count -= 1

    def get_tree(self) -> Dict[str, List[str]]:
        """Return the delegation tree (parent -> child delegation IDs)."""
        return dict(self._tree)

    def get_record(self, delegation_id: str) -> Optional[DelegationRecord]:
        """Get the record for a specific delegation."""
        return self._records.get(delegation_id)

    def get_all_records(self) -> Dict[str, DelegationRecord]:
        """Get all delegation records."""
        return dict(self._records)

    @property
    def active_count(self) -> int:
        """Number of currently running delegations."""
        return self._active_count
