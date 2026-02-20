"""Tests for dytopo.governance stalemate detection."""

import pytest

from dytopo.governance import (
    StalemateDetector,
    StalemateResult,
    get_generalist_fallback_agent,
)


class TestStalemateDetector:
    def test_no_stalemate_with_few_rounds(self):
        sd = StalemateDetector(max_ping_pong=3, max_no_progress_rounds=2)
        sd.record_round([("A", "B", 0.7)], 0.5)
        result = sd.detect()
        assert not result.is_stalled

    def test_ping_pong_detection(self):
        sd = StalemateDetector(max_ping_pong=3)
        # 3 rounds of A<->B bidirectional routing
        for _ in range(3):
            sd.record_round([("A", "B", 0.8), ("B", "A", 0.7)], 0.5)
        result = sd.detect()
        assert result.is_stalled
        assert result.suggested_action == "generalist"
        assert result.stale_pair is not None
        assert set(result.stale_pair) == {"A", "B"}

    def test_no_progress_detection(self):
        sd = StalemateDetector(max_no_progress_rounds=2)
        # 3 rounds with identical convergence score
        sd.record_round([("A", "B", 0.6)], 0.5)
        sd.record_round([("A", "B", 0.6)], 0.505)
        sd.record_round([("A", "B", 0.6)], 0.508)
        result = sd.detect()
        assert result.is_stalled
        assert result.suggested_action == "force_terminate"

    def test_regression_detection(self):
        sd = StalemateDetector()
        sd.record_round([("A", "B", 0.6)], 0.8)
        sd.record_round([("A", "B", 0.6)], 0.6)
        sd.record_round([("A", "B", 0.6)], 0.4)
        result = sd.detect()
        assert result.is_stalled
        assert result.suggested_action == "human_in_loop"

    def test_healthy_progress(self):
        sd = StalemateDetector(max_ping_pong=3, max_no_progress_rounds=2)
        sd.record_round([("A", "B", 0.6)], 0.3)
        sd.record_round([("A", "C", 0.7)], 0.5)
        sd.record_round([("B", "C", 0.8)], 0.7)
        result = sd.detect()
        assert not result.is_stalled

    def test_ping_pong_not_triggered_without_enough_rounds(self):
        sd = StalemateDetector(max_ping_pong=3)
        sd.record_round([("A", "B", 0.8), ("B", "A", 0.7)], 0.5)
        sd.record_round([("A", "B", 0.8), ("B", "A", 0.7)], 0.5)
        # Only 2 rounds, need 3
        result = sd.detect()
        assert not result.is_stalled


class TestGeneralistFallback:
    def test_selects_least_used(self):
        states = {
            "A": {"metrics": {"total_rounds": 5}},
            "B": {"metrics": {"total_rounds": 3}},
            "C": {"metrics": {"total_rounds": 1}},
        }
        result = get_generalist_fallback_agent(states, ("A", "B"))
        assert result == "C"

    def test_no_candidates(self):
        states = {
            "A": {"metrics": {"total_rounds": 5}},
            "B": {"metrics": {"total_rounds": 3}},
        }
        result = get_generalist_fallback_agent(states, ("A", "B"))
        assert result is None

    def test_empty_states(self):
        result = get_generalist_fallback_agent({}, None)
        assert result is None

    def test_no_stale_pair(self):
        states = {
            "A": {"metrics": {"total_rounds": 5}},
            "B": {"metrics": {"total_rounds": 1}},
        }
        result = get_generalist_fallback_agent(states, None)
        assert result == "B"
