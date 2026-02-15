"""Tests for DyTopo performance safeguards."""

import asyncio
import time

import pytest

from dytopo.safeguards import (
    CircuitBreaker,
    CircuitBreakerOpen,
    PerformanceSafeguards,
    RateLimiter,
    TokenBudget,
    TokenBudgetExceeded,
)


# ── RateLimiter ──────────────────────────────────────────────────────────────


class TestRateLimiter:
    def test_allows_requests_under_limit(self):
        limiter = RateLimiter(max_requests_per_minute=10)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(5):
                loop.run_until_complete(limiter.acquire())
            assert limiter.current_rate == 5
        finally:
            loop.close()

    def test_tracks_current_rate(self):
        limiter = RateLimiter(max_requests_per_minute=100)
        loop = asyncio.new_event_loop()
        try:
            for _ in range(7):
                loop.run_until_complete(limiter.acquire())
            assert limiter.current_rate == 7
        finally:
            loop.close()

    def test_rate_limit_blocks_when_exceeded(self):
        """With a limit of 2/min, third request should cause a delay."""
        limiter = RateLimiter(max_requests_per_minute=2)
        loop = asyncio.new_event_loop()
        try:
            # Fill the window
            loop.run_until_complete(limiter.acquire())
            loop.run_until_complete(limiter.acquire())
            assert limiter.current_rate == 2

            # Manually expire the oldest timestamp to avoid a 60s wait
            limiter._timestamps[0] = time.monotonic() - 61.0

            # Now acquire should succeed quickly (oldest expired)
            start = time.monotonic()
            loop.run_until_complete(limiter.acquire())
            elapsed = time.monotonic() - start
            assert elapsed < 1.0  # Should not block significantly
        finally:
            loop.close()


# ── TokenBudget ──────────────────────────────────────────────────────────────


class TestTokenBudget:
    def test_allows_within_budget(self):
        budget = TokenBudget(total_budget=1000)
        budget.check(500)  # should not raise
        budget.consume(500)
        assert budget.consumed == 500
        assert budget.remaining == 500

    def test_rejects_over_budget(self):
        budget = TokenBudget(total_budget=1000)
        budget.consume(800)
        with pytest.raises(TokenBudgetExceeded):
            budget.check(300)  # 800 + 300 > 1000

    def test_exact_budget_allowed(self):
        budget = TokenBudget(total_budget=1000)
        budget.consume(500)
        budget.check(500)  # exactly at limit, should not raise

    def test_remaining_never_negative(self):
        budget = TokenBudget(total_budget=100)
        budget.consume(150)  # over-consume
        assert budget.remaining == 0

    def test_reset(self):
        budget = TokenBudget(total_budget=1000)
        budget.consume(800)
        budget.reset()
        assert budget.consumed == 0
        assert budget.remaining == 1000


# ── CircuitBreaker ───────────────────────────────────────────────────────────


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitBreaker.CLOSED
        cb.check()  # should not raise

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED

        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitBreaker.OPEN

        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # should reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED  # only 2 consecutive failures

    def test_auto_reset_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        # Wait for reset timeout
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.check()  # should NOT raise in half-open

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED


# ── PerformanceSafeguards (composite) ────────────────────────────────────────


class TestPerformanceSafeguards:
    def test_pre_request_checks_all(self):
        safeguards = PerformanceSafeguards(
            rate_limiter=RateLimiter(max_requests_per_minute=100),
            token_budget=TokenBudget(total_budget=10000),
            circuit_breaker=CircuitBreaker(failure_threshold=5),
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(safeguards.pre_request(estimated_tokens=500))
        finally:
            loop.close()

    def test_pre_request_fails_on_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        safeguards = PerformanceSafeguards(circuit_breaker=cb)

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(CircuitBreakerOpen):
                loop.run_until_complete(safeguards.pre_request())
        finally:
            loop.close()

    def test_pre_request_fails_on_budget_exceeded(self):
        budget = TokenBudget(total_budget=100)
        budget.consume(90)
        safeguards = PerformanceSafeguards(token_budget=budget)

        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(TokenBudgetExceeded):
                loop.run_until_complete(safeguards.pre_request(estimated_tokens=50))
        finally:
            loop.close()

    def test_post_request_tracks_tokens_and_success(self):
        safeguards = PerformanceSafeguards(
            token_budget=TokenBudget(total_budget=10000),
            circuit_breaker=CircuitBreaker(failure_threshold=3),
        )
        safeguards.post_request(tokens_used=500, success=True)
        assert safeguards.token_budget.consumed == 500
        assert safeguards.circuit_breaker.state == CircuitBreaker.CLOSED

    def test_post_request_tracks_failure(self):
        cb = CircuitBreaker(failure_threshold=2)
        safeguards = PerformanceSafeguards(circuit_breaker=cb)
        safeguards.post_request(tokens_used=100, success=False)
        safeguards.post_request(tokens_used=100, success=False)
        assert cb.state == CircuitBreaker.OPEN

    def test_default_construction(self):
        """PerformanceSafeguards can be created with no arguments."""
        safeguards = PerformanceSafeguards()
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(safeguards.pre_request())
            safeguards.post_request(tokens_used=100, success=True)
        finally:
            loop.close()
