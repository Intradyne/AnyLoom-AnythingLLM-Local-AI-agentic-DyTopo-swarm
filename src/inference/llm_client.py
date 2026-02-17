"""Async inference client for llama.cpp backend.

Includes health checks, retry logic, token tracking, and concurrency control.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class Backend(Enum):
    """Supported inference backends."""
    LLAMA_CPP = "llama-cpp"


@dataclass
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class CompletionResult:
    """Result from a chat completion request."""
    content: str
    parsed: Optional[Any] = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    backend: Backend = Backend.LLAMA_CPP


class InferenceClient:
    """Async inference client with concurrency control.

    Features:
    - Health checks via /v1/models endpoint
    - Retry logic with exponential backoff
    - Per-agent token tracking
    - Semaphore-controlled concurrency
    - Connection pooling via httpx
    """

    def __init__(self, config: dict):
        """Initialize the inference client.

        Args:
            config: Configuration dict with keys:
                - base_url: API base URL (default: http://localhost:8008/v1)
                - api_key: API key (default: "EMPTY")
                - model: Model name (default: "meta-llama/Llama-3.1-8B-Instruct")
                - max_concurrent: Max concurrent requests (optional, auto-detected)
        """
        self.base_url = config.get("base_url", "http://localhost:8008/v1")
        self.api_key = config.get("api_key", "EMPTY")
        self.model = config.get("model", "meta-llama/Llama-3.1-8B-Instruct")

        # Initialize OpenAI client with httpx for connection pooling
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            http_client=httpx.AsyncClient(
                limits=httpx.Limits(
                    max_keepalive_connections=20,
                    max_connections=100
                ),
                timeout=httpx.Timeout(60.0)
            )
        )

        # Backend detection (will be set during health check)
        self.backend: Optional[Backend] = None

        # Concurrency control
        self._max_concurrent = config.get("max_concurrent")
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Token tracking per agent
        self._token_usage: dict[str, TokenUsage] = {}
        self._usage_lock = asyncio.Lock()

    async def health_check(self) -> dict:
        """Check backend health and detect backend type.

        Returns:
            dict: Health check result with keys:
                - status: "healthy" or "unhealthy"
                - backend: Backend enum
                - models: List of available models
                - error: Error message (if unhealthy)

        Raises:
            httpx.HTTPError: If health check fails
        """
        try:
            # Use httpx to GET /v1/models
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Remove /v1 suffix if present to construct models URL
                base = self.base_url.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                models_url = f"{base}/v1/models"

                response = await client.get(models_url)
                response.raise_for_status()
                data = response.json()

                models = data.get("data", [])
                self.backend = Backend.LLAMA_CPP

                # Set concurrency limit
                if self._max_concurrent is None:
                    self._max_concurrent = 2
                self._semaphore = asyncio.Semaphore(self._max_concurrent)

                logger.info(
                    f"Health check passed: backend={self.backend.value}, "
                    f"max_concurrent={self._max_concurrent}, models={len(models)}"
                )

                return {
                    "status": "healthy",
                    "backend": self.backend,
                    "models": [m.get("id") for m in models],
                    "max_concurrent": self._max_concurrent
                }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.backend = Backend.LLAMA_CPP
            if self._max_concurrent is None:
                self._max_concurrent = 2
            self._semaphore = asyncio.Semaphore(self._max_concurrent)

            return {
                "status": "unhealthy",
                "backend": self.backend,
                "models": [],
                "error": str(e)
            }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8), reraise=True)
    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        agent_id: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
        timeout: float = 120.0,
        **kwargs
    ) -> CompletionResult:
        """Execute a chat completion request with retry logic.

        Args:
            messages: List of message dicts with "role" and "content"
            agent_id: Agent identifier for token tracking
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            response_format: Optional response format (e.g., {"type": "json_object"})
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            CompletionResult: Completion result with content and metadata

        Raises:
            Exception: If all retry attempts fail
        """
        # Ensure health check has been run
        if self.backend is None or self._semaphore is None:
            await self.health_check()

        start_time = time.perf_counter()

        # Acquire semaphore for concurrency control
        async with self._semaphore:
            try:
                # Build request parameters
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }

                if response_format:
                    params["response_format"] = response_format

                # Make API call with wall-clock timeout
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(**params),
                    timeout=timeout,
                )

                # Extract data
                content = response.choices[0].message.content or ""
                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0

                # Calculate latency
                latency_ms = (time.perf_counter() - start_time) * 1000

                # Update token tracking
                async with self._usage_lock:
                    if agent_id not in self._token_usage:
                        self._token_usage[agent_id] = TokenUsage()

                    usage = self._token_usage[agent_id]
                    usage.prompt_tokens += tokens_in
                    usage.completion_tokens += tokens_out
                    usage.total_tokens += (tokens_in + tokens_out)

                logger.debug(
                    f"Completion for {agent_id}: {tokens_in} in, {tokens_out} out, "
                    f"{latency_ms:.1f}ms ({self.backend.value})"
                )

                return CompletionResult(
                    content=content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    latency_ms=latency_ms,
                    backend=self.backend
                )

            except Exception as e:
                logger.error(f"Completion failed for {agent_id}: {e}")
                raise

    async def batch_completions(
        self,
        requests: list[dict[str, Any]]
    ) -> list[CompletionResult]:
        """Execute multiple completion requests concurrently.

        Args:
            requests: List of request dicts, each containing:
                - messages: List of message dicts
                - agent_id: Agent identifier (optional)
                - temperature: Sampling temperature (optional)
                - max_tokens: Max tokens (optional)
                - response_format: Response format (optional)
                - Any other kwargs for chat_completion

        Returns:
            list[CompletionResult]: List of completion results
        """
        tasks = []
        for req in requests:
            # Extract parameters
            messages = req.pop("messages")
            agent_id = req.pop("agent_id", "default")
            temperature = req.pop("temperature", 0.7)
            max_tokens = req.pop("max_tokens", 2048)
            response_format = req.pop("response_format", None)

            # Create task
            task = self.chat_completion(
                messages=messages,
                agent_id=agent_id,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **req
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error results
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch completion failed: {result}")
                final_results.append(
                    CompletionResult(
                        content=f"Error: {str(result)}",
                        backend=self.backend or Backend.LLAMA_CPP
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def get_token_usage(self, agent_id: str = "default") -> TokenUsage:
        """Get token usage statistics for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            TokenUsage: Token usage statistics (returns empty if not found)
        """
        return self._token_usage.get(agent_id, TokenUsage())

    def reset_token_usage(self, agent_id: Optional[str] = None):
        """Reset token usage statistics.

        Args:
            agent_id: Agent identifier (if None, resets all)
        """
        if agent_id is None:
            self._token_usage.clear()
            logger.info("Reset token usage for all agents")
        elif agent_id in self._token_usage:
            del self._token_usage[agent_id]
            logger.info(f"Reset token usage for {agent_id}")

    async def close(self):
        """Close the client and cleanup resources."""
        if self.client and hasattr(self.client, 'close'):
            await self.client.close()
        logger.info("Inference client closed")


# Singleton pattern for global client instance
_global_client: Optional[InferenceClient] = None
_client_lock = asyncio.Lock()


async def get_client(config: Optional[dict] = None) -> InferenceClient:
    """Get or create the global inference client (singleton).

    Args:
        config: Configuration dict (only used on first call)

    Returns:
        InferenceClient: Global client instance
    """
    global _global_client

    async with _client_lock:
        if _global_client is None:
            if config is None:
                config = {}
            _global_client = InferenceClient(config)
            await _global_client.health_check()
            logger.info("Created global inference client")

        return _global_client


async def reset_client():
    """Reset the global client (useful for testing)."""
    global _global_client

    async with _client_lock:
        if _global_client is not None:
            await _global_client.close()
            _global_client = None
            logger.info("Reset global inference client")
