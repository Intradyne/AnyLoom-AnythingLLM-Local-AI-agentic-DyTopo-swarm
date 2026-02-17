"""
Performance Safeguards for DyTopo Swarms
=========================================

Rate limiting, token budget enforcement, and circuit breaker pattern
to prevent runaway consumption and cascading failures.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List

logger = logging.getLogger("dytopo.safeguards")


class TokenBudgetExceeded(Exception):
    """Raised when token budget would be exceeded."""


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open due to repeated failures."""


class RateLimiter:
    """Token-bucket rate limiter for requests per minute.

    Tracks request timestamps and sleeps if the rate would be exceeded.
    Supports burst mode for parallel execution.
    """

    def __init__(self, max_requests_per_minute: int = 120, burst_size: int = 0):
        self.max_rpm = max_requests_per_minute
        self._timestamps: List[float] = []
        self.burst_size = burst_size  # allow this many requests immediately before rate-limiting
        self._burst_remaining = burst_size

    async def acquire(self) -> None:
        """Acquire permission to make a request. Sleeps if rate limit hit.

        Supports burst mode: first N requests (burst_size) are allowed immediately.
        """
        # Check burst allowance first
        if self._burst_remaining > 0:
            self._burst_remaining -= 1
            self._timestamps.append(time.monotonic())
            return

        now = time.monotonic()

        # Remove timestamps older than 60 seconds
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]

        if len(self._timestamps) >= self.max_rpm:
            # Must wait until oldest request falls out of the window
            oldest = self._timestamps[0]
            wait_time = 60.0 - (now - oldest)
            if wait_time > 0:
                logger.warning(f"Rate limit reached ({self.max_rpm}/min), waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)

        self._timestamps.append(time.monotonic())

    @property
    def current_rate(self) -> int:
        """Number of requests in the current 60-second window."""
        now = time.monotonic()
        cutoff = now - 60.0
        return sum(1 for t in self._timestamps if t > cutoff)


class TokenBudget:
    """Enforces a total token consumption budget.

    Tracks tokens consumed and raises TokenBudgetExceeded if a request
    would push consumption over the limit.
    """

    def __init__(self, total_budget: int = 100_000):
        self.total_budget = total_budget
        self._consumed: int = 0

    def check(self, tokens_needed: int) -> None:
        """Check if tokens_needed fits within remaining budget.

        Raises:
            TokenBudgetExceeded: If budget would be exceeded.
        """
        if self._consumed + tokens_needed > self.total_budget:
            raise TokenBudgetExceeded(
                f"Token budget exhausted. "
                f"Budget: {self.total_budget}, used: {self._consumed}, "
                f"needed: {tokens_needed}, remaining: {self.remaining}"
            )

    def consume(self, tokens: int) -> None:
        """Record token consumption."""
        self._consumed += tokens

    @property
    def consumed(self) -> int:
        return self._consumed

    @property
    def remaining(self) -> int:
        return max(0, self.total_budget - self._consumed)

    def reset(self) -> None:
        """Reset consumed tokens to zero."""
        self._consumed = 0


class CircuitBreaker:
    """Circuit breaker with closed/open/half-open states.

    Opens after `failure_threshold` consecutive failures, preventing
    further requests until `reset_timeout` elapses. After timeout,
    enters half-open state: one request allowed to test recovery.

    Backend-aware: llama-cpp has higher tolerance for transient errors.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        backend: str = "llama-cpp"
    ):
        # llama-cpp: higher threshold (more tolerant of transient errors during batch inference)
        if backend == "llama-cpp":
            failure_threshold = max(failure_threshold, 8)

        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.backend = backend
        self._state: str = self.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: float = 0.0

    @property
    def state(self) -> str:
        # Auto-transition from OPEN to HALF_OPEN after timeout
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.reset_timeout:
                self._state = self.HALF_OPEN
                logger.info("Circuit breaker transitioning to half-open")
        return self._state

    def check(self) -> None:
        """Check if requests are allowed.

        Raises:
            CircuitBreakerOpen: If circuit is open and timeout hasn't elapsed.
        """
        current = self.state  # triggers auto-transition
        if current == self.OPEN:
            remaining = self.reset_timeout - (time.monotonic() - self._last_failure_time)
            raise CircuitBreakerOpen(
                f"Circuit breaker open after {self._failure_count} failures. "
                f"Retry in {remaining:.1f}s"
            )
        # CLOSED and HALF_OPEN allow requests

    def record_success(self) -> None:
        """Record a successful request. Resets failure count and closes circuit."""
        self._failure_count = 0
        if self._state != self.CLOSED:
            logger.info(f"Circuit breaker closed after success (was {self._state})")
        self._state = self.CLOSED

    def record_failure(self) -> None:
        """Record a failed request. May open the circuit."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._failure_count >= self.failure_threshold:
            if self._state != self.OPEN:
                logger.error(
                    f"Circuit breaker opened after {self._failure_count} consecutive failures"
                )
            self._state = self.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self._state = self.CLOSED
        self._failure_count = 0


class PerformanceSafeguards:
    """Composite facade wrapping rate limiter, token budget, and circuit breaker.

    Usage:
        safeguards = PerformanceSafeguards()
        await safeguards.pre_request(estimated_tokens=500)
        # ... make request ...
        safeguards.post_request(tokens_used=450, success=True)
    """

    def __init__(
        self,
        rate_limiter: RateLimiter | None = None,
        token_budget: TokenBudget | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.token_budget = token_budget or TokenBudget()
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    async def pre_request(self, estimated_tokens: int = 0) -> None:
        """Run all pre-request checks. Raises on failure.

        Checks circuit breaker, token budget, and rate limit (in that order).
        """
        self.circuit_breaker.check()
        if estimated_tokens > 0:
            self.token_budget.check(estimated_tokens)
        await self.rate_limiter.acquire()

    def post_request(self, tokens_used: int, success: bool) -> None:
        """Record request outcome for all safeguards."""
        self.token_budget.consume(tokens_used)
        if success:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()
