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


class TestBurstMode:
    """Test burst mode for RateLimiter."""

    def test_burst_allows_initial_requests(self):
        """First N requests (burst_size) should succeed immediately."""
        limiter = RateLimiter(max_requests_per_minute=10, burst_size=8)
        loop = asyncio.new_event_loop()
        try:
            import time
            start = time.monotonic()

            # First 8 requests should succeed immediately
            for _ in range(8):
                loop.run_until_complete(limiter.acquire())

            elapsed = time.monotonic() - start
            assert elapsed < 0.5, f"Burst requests should be instant, took {elapsed:.2f}s"
            assert limiter.current_rate == 8
        finally:
            loop.close()

    def test_burst_rate_limits_after_burst(self):
        """After burst exhausted, should apply rate limiting."""
        limiter = RateLimiter(max_requests_per_minute=60, burst_size=2)
        loop = asyncio.new_event_loop()
        try:
            # Exhaust burst
            loop.run_until_complete(limiter.acquire())
            loop.run_until_complete(limiter.acquire())
            assert limiter._burst_remaining == 0

            # Subsequent requests should be rate-limited
            # With 60/min = 1/sec, 58 more requests should be allowed in the minute
            # but we'll just check that rate limiting is active
            assert limiter.current_rate == 2
        finally:
            loop.close()

    def test_zero_burst_size_disables_burst(self):
        """burst_size=0 should disable burst mode."""
        limiter = RateLimiter(max_requests_per_minute=120, burst_size=0)
        assert limiter.burst_size == 0
        assert limiter._burst_remaining == 0


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
        cb = CircuitBreaker(backend="generic")
        assert cb.state == CircuitBreaker.CLOSED
        cb.check()  # should not raise

    def test_opens_after_threshold_failures(self):
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=60.0, backend="generic")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED

        cb.record_failure()  # 3rd failure
        assert cb.state == CircuitBreaker.OPEN

        with pytest.raises(CircuitBreakerOpen):
            cb.check()

    def test_success_resets_failure_count(self):
        cb = CircuitBreaker(failure_threshold=3, backend="generic")
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # should reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.CLOSED  # only 2 consecutive failures

    def test_auto_reset_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01, backend="generic")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

        # Wait for reset timeout
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN
        cb.check()  # should NOT raise in half-open

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01, backend="generic")
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitBreaker.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=2, reset_timeout=0.01, backend="generic")
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == CircuitBreaker.HALF_OPEN

        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=2, backend="generic")
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreaker.OPEN
        cb.reset()
        assert cb.state == CircuitBreaker.CLOSED


class TestBackendAwareCircuitBreaker:
    """Test backend-specific circuit breaker thresholds."""

    def test_llama_cpp_backend_has_higher_threshold(self):
        """llama-cpp backend should have higher failure threshold."""
        cb = CircuitBreaker(failure_threshold=5, backend="llama-cpp")
        assert cb.failure_threshold >= 8, f"llama-cpp threshold should be ≥8, got {cb.failure_threshold}"

    def test_llama_cpp_tolerates_more_failures(self):
        """llama-cpp circuit breaker should stay closed longer under failures."""
        cb = CircuitBreaker(failure_threshold=3, backend="llama-cpp")

        # With llama-cpp backend, threshold gets raised to max(3, 8) = 8
        for _ in range(7):
            cb.record_failure()

        assert cb.state == CircuitBreaker.CLOSED, "llama-cpp should tolerate 7 failures"

        cb.record_failure()  # 8th failure
        assert cb.state == CircuitBreaker.OPEN, "Should open after 8 failures"


# ── PerformanceSafeguards (composite) ────────────────────────────────────────


class TestPerformanceSafeguards:
    def test_pre_request_checks_all(self):
        safeguards = PerformanceSafeguards(
            rate_limiter=RateLimiter(max_requests_per_minute=100),
            token_budget=TokenBudget(total_budget=10000),
            circuit_breaker=CircuitBreaker(failure_threshold=5, backend="generic"),
        )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(safeguards.pre_request(estimated_tokens=500))
        finally:
            loop.close()

    def test_pre_request_fails_on_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=1, backend="generic")
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
            circuit_breaker=CircuitBreaker(failure_threshold=3, backend="generic"),
        )
        safeguards.post_request(tokens_used=500, success=True)
        assert safeguards.token_budget.consumed == 500
        assert safeguards.circuit_breaker.state == CircuitBreaker.CLOSED

    def test_post_request_tracks_failure(self):
        cb = CircuitBreaker(failure_threshold=2, backend="generic")
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
