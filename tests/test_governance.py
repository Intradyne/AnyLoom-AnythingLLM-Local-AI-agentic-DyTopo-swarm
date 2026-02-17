"""
Tests for dytopo.governance module
===================================

Verifies failure handling, convergence detection, and re-delegation logic.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dytopo.governance import (
    execute_agent_safe,
    detect_convergence,
    detect_stalling,
    recommend_redelegation,
    update_agent_metrics,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TEST HELPERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def mock_successful_agent():
    """Simulates a successful agent call."""
    await asyncio.sleep(0.01)
    return {
        "key": "Implemented function X",
        "query": "Need test cases from tester",
        "work": "def hello(): return 'world'",
    }


async def mock_failing_agent():
    """Simulates an agent that always fails."""
    await asyncio.sleep(0.01)
    raise RuntimeError("Agent encountered an error")


async def mock_timeout_agent():
    """Simulates an agent that times out."""
    await asyncio.sleep(10)
    return {"key": "Done", "query": "", "work": "Late result"}


async def mock_invalid_json_agent():
    """Simulates an agent returning invalid JSON."""
    await asyncio.sleep(0.01)
    return "This is not JSON at all!"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TESTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.asyncio
async def test_execute_agent_safe_success():
    """Test successful agent execution."""
    print("\n[TEST] execute_agent_safe with successful agent")
    # Pass callable so retries work
    result = await execute_agent_safe("dev", mock_successful_agent, timeout_sec=1.0)
    assert result["success"] is True, "Should succeed"
    assert "Implemented function X" in result["key"], "Should have key field"
    assert result["retries"] == 0, "Should not retry on success"
    print("[PASS] Successful agent execution works")


@pytest.mark.asyncio
async def test_execute_agent_safe_timeout():
    """Test timeout handling."""
    print("\n[TEST] execute_agent_safe with timeout")
    # Pass callable for retry support
    result = await execute_agent_safe("slow", mock_timeout_agent, timeout_sec=0.5, max_retries=1)
    assert result["success"] is False, "Should fail on timeout"
    assert "Timeout" in result.get("error", "") or "timeout" in result.get("error", "").lower(), \
        f"Should report timeout, got: {result.get('error', '')}"
    assert result["retries"] > 0, "Should have attempted retries"
    print("[PASS] Timeout handling works")


@pytest.mark.asyncio
async def test_execute_agent_safe_failure():
    """Test exception handling."""
    print("\n[TEST] execute_agent_safe with failing agent")
    # Pass callable for retry support
    result = await execute_agent_safe("broken", mock_failing_agent, timeout_sec=1.0, max_retries=1)
    assert result["success"] is False, "Should fail on exception"
    assert "error" in result, "Should have error field"
    assert "[FAILURE]" in result["key"], "Should return failure stub"
    print("[PASS] Exception handling works")


@pytest.mark.asyncio
async def test_execute_agent_safe_invalid_json():
    """Test invalid JSON handling."""
    print("\n[TEST] execute_agent_safe with invalid JSON")
    result = await execute_agent_safe("malformed", mock_invalid_json_agent, timeout_sec=1.0, max_retries=0)
    # Should either parse it or fail gracefully
    assert isinstance(result, dict), "Should return a dict"
    print("[PASS] Invalid JSON handling works")


def test_detect_convergence():
    """Test convergence detection."""
    print("\n[TEST] detect_convergence")

    # Non-converged rounds (different outputs)
    round_history = [
        {"round": 1, "outputs": {"dev": {"work": "version 1"}}},
        {"round": 2, "outputs": {"dev": {"work": "version 2"}}},
        {"round": 3, "outputs": {"dev": {"work": "version 3"}}},
    ]
    converged, similarity = detect_convergence(round_history, window_size=3, similarity_threshold=0.95)
    assert not converged, "Should not detect convergence for different outputs"
    print(f"  Non-converged rounds: similarity={similarity:.2%}")

    # Converged rounds (identical outputs)
    round_history = [
        {"round": 1, "outputs": {"dev": {"work": "final version"}}},
        {"round": 2, "outputs": {"dev": {"work": "final version"}}},
        {"round": 3, "outputs": {"dev": {"work": "final version"}}},
    ]
    converged, similarity = detect_convergence(round_history, window_size=3, similarity_threshold=0.95)
    assert converged, "Should detect convergence for identical outputs"
    assert similarity >= 0.95, f"Similarity should be high, got {similarity}"
    print(f"  Converged rounds: similarity={similarity:.2%}")
    print("[PASS] Convergence detection works")


def test_detect_stalling():
    """Test agent stalling detection."""
    print("\n[TEST] detect_stalling")

    # Agent producing varied outputs
    round_history = [
        {"round": 1, "outputs": {"dev": {"work": "attempt 1"}}},
        {"round": 2, "outputs": {"dev": {"work": "attempt 2"}}},
        {"round": 3, "outputs": {"dev": {"work": "attempt 3"}}},
    ]
    stalling, similarity = detect_stalling("dev", round_history, window_size=3, similarity_threshold=0.98)
    assert not stalling, "Should not detect stalling for varied outputs"
    print(f"  Varied outputs: similarity={similarity:.2%}")

    # Agent producing identical outputs
    round_history = [
        {"round": 1, "outputs": {"dev": {"work": "stuck response"}}},
        {"round": 2, "outputs": {"dev": {"work": "stuck response"}}},
        {"round": 3, "outputs": {"dev": {"work": "stuck response"}}},
    ]
    stalling, similarity = detect_stalling("dev", round_history, window_size=3, similarity_threshold=0.98)
    assert stalling, "Should detect stalling for identical outputs"
    assert similarity >= 0.98, f"Similarity should be very high, got {similarity}"
    print(f"  Identical outputs: similarity={similarity:.2%}")
    print("[PASS] Stalling detection works")


def test_recommend_redelegation():
    """Test re-delegation recommendations."""
    print("\n[TEST] recommend_redelegation")

    # Create history with a stalling agent and a failing agent
    round_history = [
        {
            "round": 1,
            "outputs": {
                "dev": {"work": "same output", "success": True},
                "tester": {"work": "fail", "success": False},
            },
            "edges": [{"source": "dev", "target": "tester"}],
        },
        {
            "round": 2,
            "outputs": {
                "dev": {"work": "same output", "success": True},
                "tester": {"work": "fail", "success": False},
            },
            "edges": [{"source": "dev", "target": "tester"}],
        },
        {
            "round": 3,
            "outputs": {
                "dev": {"work": "same output", "success": True},
                "tester": {"work": "fail", "success": False},
            },
            "edges": [],
        },
    ]

    agent_roles = {
        "dev": {"id": "dev", "name": "Developer"},
        "tester": {"id": "tester", "name": "Tester"},
    }

    recommendations = recommend_redelegation(
        round_history, agent_roles, stall_threshold=0.98, failure_threshold=2
    )

    print(f"  Generated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"    - {rec['agent_id']} [{rec['severity']}]: {rec['reason'][:50]}...")

    # Should recommend re-delegation for both agents
    agent_ids_with_recs = {rec["agent_id"] for rec in recommendations}
    assert "dev" in agent_ids_with_recs, "Should recommend for stalling dev agent"
    assert "tester" in agent_ids_with_recs, "Should recommend for failing tester agent"
    print("[PASS] Re-delegation recommendations work")


def test_update_agent_metrics():
    """Test metrics tracking."""
    print("\n[TEST] update_agent_metrics")

    agent_state = {"id": "dev", "failure_count": 0}

    # Note: update_agent_metrics is now async, but for this test we need to handle it
    loop = asyncio.new_event_loop()
    try:
        # Successful round
        loop.run_until_complete(update_agent_metrics(agent_state, {"success": True, "retries": 0}))
        assert agent_state["metrics"]["total_rounds"] == 1
        assert agent_state["metrics"]["total_failures"] == 0
        print(f"  After success: {agent_state['metrics']}")

        # Failed round
        loop.run_until_complete(update_agent_metrics(agent_state, {"success": False, "retries": 2, "error": "Timeout"}))
        assert agent_state["metrics"]["total_rounds"] == 2
        assert agent_state["metrics"]["total_failures"] == 1
        assert agent_state["failure_count"] == 1
        assert agent_state["metrics"]["failure_rate"] == 0.5
        print(f"  After failure: {agent_state['metrics']}")
        print("[PASS] Metrics tracking works")
    finally:
        loop.close()


class TestBackendRetryPolicy:
    """Test backend-aware retry policies."""

    def test_llama_cpp_policy_fast_recovery(self):
        """llama-cpp policy should have shorter delays for fast recovery."""
        from dytopo.governance import BackendRetryPolicy

        policy = BackendRetryPolicy.for_backend("llama-cpp")
        assert policy.base_delay <= 1.0, f"llama-cpp base delay should be ≤1s, got {policy.base_delay}"
        assert policy.max_delay <= 4.0, f"llama-cpp max delay should be ≤4s, got {policy.max_delay}"
        assert policy.jitter is True, "llama-cpp should use jitter"

    def test_unknown_backend_uses_llama_cpp_defaults(self):
        """Any backend string should return the llama-cpp policy."""
        from dytopo.governance import BackendRetryPolicy

        policy = BackendRetryPolicy.for_backend("unknown_backend")
        assert policy.base_delay <= 1.0
        assert policy.max_delay <= 4.0

    def test_calculate_delay_increases_with_attempts(self):
        """Delay should increase exponentially with attempt number."""
        from dytopo.governance import BackendRetryPolicy

        policy = BackendRetryPolicy.for_backend("llama-cpp")
        delay_0 = policy.calculate_delay(0)
        delay_1 = policy.calculate_delay(1)
        delay_2 = policy.calculate_delay(2)

        # Should be roughly: base * 2^0, base * 2^1, base * 2^2 (with jitter)
        assert delay_1 > delay_0, "Delay should increase with attempts"
        assert delay_2 > delay_1, "Delay should increase with attempts"

    def test_delay_respects_max_delay(self):
        """Delay should never exceed max_delay."""
        from dytopo.governance import BackendRetryPolicy

        policy = BackendRetryPolicy.for_backend("llama-cpp")
        for attempt in range(10):
            delay = policy.calculate_delay(attempt)
            assert delay <= policy.max_delay * 1.5, f"Delay {delay} exceeds max {policy.max_delay} (with jitter)"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def run_async_tests():
    """Run all async tests."""
    await test_execute_agent_safe_success()
    await test_execute_agent_safe_timeout()
    await test_execute_agent_safe_failure()
    await test_execute_agent_safe_invalid_json()


def run_sync_tests():
    """Run all sync tests."""
    test_detect_convergence()
    test_detect_stalling()
    test_recommend_redelegation()
    test_update_agent_metrics()


if __name__ == "__main__":
    print("=" * 70)
    print("DyTopo Governance Tests")
    print("=" * 70)

    # Run sync tests
    run_sync_tests()

    # Run async tests
    asyncio.run(run_async_tests())

    print("\n" + "=" * 70)
    print("All tests passed! [PASS]")
    print("=" * 70)
