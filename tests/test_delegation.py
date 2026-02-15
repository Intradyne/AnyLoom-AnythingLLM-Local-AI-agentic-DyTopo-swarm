"""Tests for DyTopo delegation system."""

import asyncio

import pytest

from dytopo.delegation import DelegationError, DelegationManager, DelegationRecord


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _mock_runner_success(subtask: str, context: dict) -> str:
    """Mock swarm runner that succeeds after a short delay."""
    await asyncio.sleep(0.01)
    return f"Completed: {subtask}"


async def _mock_runner_slow(subtask: str, context: dict) -> str:
    """Mock swarm runner that takes a long time (simulates timeout)."""
    await asyncio.sleep(10.0)
    return "should not reach here"


async def _mock_runner_failure(subtask: str, context: dict) -> str:
    """Mock swarm runner that raises an error."""
    await asyncio.sleep(0.01)
    raise RuntimeError("Runner encountered an error")


# ── DelegationRecord ─────────────────────────────────────────────────────────


class TestDelegationRecord:
    def test_default_status(self):
        record = DelegationRecord(
            delegation_id="test_001",
            parent_agent_id="developer",
            subtask="verify solution",
            depth=0,
            start_time=100.0,
        )
        assert record.status == "pending"
        assert record.result is None
        assert record.error is None
        assert record.duration is None

    def test_duration_calculation(self):
        record = DelegationRecord(
            delegation_id="test_001",
            parent_agent_id="developer",
            subtask="verify",
            depth=0,
            start_time=100.0,
            end_time=105.5,
        )
        assert record.duration == pytest.approx(5.5)


# ── DelegationManager ───────────────────────────────────────────────────────


class TestDelegationManager:
    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_single_delegation_success(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        result = self._run(manager.delegate(
            parent_agent_id="developer",
            subtask="verify the proof",
            context={"domain": "math"},
            swarm_runner=_mock_runner_success,
        ))
        assert result == "Completed: verify the proof"
        assert manager.active_count == 0

    def test_delegation_tree_tracked(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        self._run(manager.delegate(
            "developer", "subtask_1", {}, _mock_runner_success,
        ))
        self._run(manager.delegate(
            "developer", "subtask_2", {}, _mock_runner_success,
        ))
        self._run(manager.delegate(
            "tester", "subtask_3", {}, _mock_runner_success,
        ))

        tree = manager.get_tree()
        assert len(tree["developer"]) == 2
        assert len(tree["tester"]) == 1

    def test_records_tracked(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        self._run(manager.delegate(
            "developer", "check code", {}, _mock_runner_success,
        ))
        records = manager.get_all_records()
        assert len(records) == 1
        record = list(records.values())[0]
        assert record.status == "completed"
        assert record.result == "Completed: check code"
        assert record.duration is not None
        assert record.duration > 0

    def test_max_depth_exceeded(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        with pytest.raises(DelegationError, match="depth"):
            self._run(manager.delegate(
                "developer", "too deep", {},
                _mock_runner_success,
                depth=2,  # At max depth, should fail
            ))

    def test_depth_zero_and_one_allowed(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        # depth 0 should work
        result = self._run(manager.delegate(
            "developer", "d0", {}, _mock_runner_success, depth=0,
        ))
        assert "Completed" in result
        # depth 1 should work
        result = self._run(manager.delegate(
            "developer", "d1", {}, _mock_runner_success, depth=1,
        ))
        assert "Completed" in result

    def test_timeout_handling(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=0.05)
        with pytest.raises(DelegationError, match="timed out"):
            self._run(manager.delegate(
                "developer", "slow task", {},
                _mock_runner_slow,
            ))
        # Record should reflect timeout
        records = manager.get_all_records()
        record = list(records.values())[0]
        assert record.status == "timed_out"

    def test_runner_failure_handling(self):
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)
        with pytest.raises(DelegationError, match="failed"):
            self._run(manager.delegate(
                "developer", "bad task", {},
                _mock_runner_failure,
            ))
        records = manager.get_all_records()
        record = list(records.values())[0]
        assert record.status == "failed"
        assert "error" in (record.error or "").lower()

    def test_concurrent_delegations(self):
        """Multiple delegations should run concurrently within semaphore limit."""
        manager = DelegationManager(max_depth=2, max_concurrent=4, timeout=5.0)

        async def run_concurrent():
            tasks = [
                manager.delegate(f"agent_{i}", f"task_{i}", {}, _mock_runner_success)
                for i in range(4)
            ]
            results = await asyncio.gather(*tasks)
            return results

        results = self._run(run_concurrent())
        assert len(results) == 4
        for i, r in enumerate(results):
            assert f"task_{i}" in r
        assert manager.active_count == 0

    def test_semaphore_limits_concurrency(self):
        """With max_concurrent=1, delegations should serialize."""
        manager = DelegationManager(max_depth=2, max_concurrent=1, timeout=5.0)

        call_order = []

        async def tracking_runner(subtask: str, context: dict) -> str:
            call_order.append(f"start_{subtask}")
            await asyncio.sleep(0.02)
            call_order.append(f"end_{subtask}")
            return subtask

        async def run_serial():
            tasks = [
                manager.delegate(f"agent_{i}", f"t{i}", {}, tracking_runner)
                for i in range(3)
            ]
            return await asyncio.gather(*tasks)

        self._run(run_serial())
        # With max_concurrent=1, starts and ends should not interleave
        # (each task must finish before the next starts)
        for i in range(0, len(call_order) - 1, 2):
            start = call_order[i]
            end = call_order[i + 1]
            # The same task that started should end next
            task_id = start.replace("start_", "")
            assert end == f"end_{task_id}"

    def test_active_count(self):
        """Active count should be 0 when no delegations running."""
        manager = DelegationManager()
        assert manager.active_count == 0

    def test_get_record_nonexistent(self):
        manager = DelegationManager()
        assert manager.get_record("nonexistent") is None
