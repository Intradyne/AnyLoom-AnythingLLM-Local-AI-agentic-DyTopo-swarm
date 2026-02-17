"""Test suite for async inference client.

Tests backend detection, health checks, chat completion, concurrency,
retry logic, and token tracking.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference.llm_client import (
    Backend,
    CompletionResult,
    InferenceClient,
    TokenUsage,
    get_client,
    reset_client,
)


@pytest.fixture
def mock_config():
    """Standard test configuration."""
    return {
        "base_url": "http://localhost:8008/v1",
        "api_key": "test-key",
        "model": "test-model",
    }


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response"))
    ]
    mock_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30
    )
    return mock_response


class TestClientCreation:
    """Test client initialization and configuration."""

    def test_init_with_defaults(self):
        """Test client creation with default values."""
        client = InferenceClient({})
        assert client.base_url == "http://localhost:8008/v1"
        assert client.api_key == "EMPTY"
        assert client.model == "meta-llama/Llama-3.1-8B-Instruct"
        assert client.backend is None
        assert client._semaphore is None

    def test_init_with_custom_config(self, mock_config):
        """Test client creation with custom configuration."""
        client = InferenceClient(mock_config)
        assert client.base_url == "http://localhost:8008/v1"
        assert client.api_key == "test-key"
        assert client.model == "test-model"

    def test_init_with_max_concurrent(self):
        """Test client creation with custom max_concurrent."""
        config = {"max_concurrent": 5}
        client = InferenceClient(config)
        assert client._max_concurrent == 5


class TestBackendSelection:
    """Test backend detection and configuration."""

    @pytest.mark.asyncio
    async def test_detect_llama_cpp_backend(self, mock_config):
        """Test llama.cpp backend detection."""
        client = InferenceClient(mock_config)

        mock_response = {
            "data": [
                {"id": "model1", "root": "meta-llama", "parent": None}
            ]
        }

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None
                )
            )

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert client.backend == Backend.LLAMA_CPP
            assert client._max_concurrent == 8
            assert client._semaphore._value == 8

    @pytest.mark.asyncio
    async def test_health_check_failure_keeps_llama_cpp(self, mock_config):
        """Test that health check failure still defaults to llama-cpp backend."""
        client = InferenceClient(mock_config)

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Connection failed")
            )

            result = await client.health_check()

            assert result["status"] == "unhealthy"
            assert client.backend == Backend.LLAMA_CPP
            assert client._max_concurrent == 8
            assert "error" in result


class TestChatCompletion:
    """Test chat completion functionality."""

    @pytest.mark.asyncio
    async def test_basic_completion(self, mock_config, mock_openai_response):
        """Test basic chat completion."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            messages = [{"role": "user", "content": "Hello"}]
            result = await client.chat_completion(messages, agent_id="test-agent")

            assert isinstance(result, CompletionResult)
            assert result.content == "Test response"
            assert result.tokens_in == 10
            assert result.tokens_out == 20
            assert result.backend == Backend.LLAMA_CPP
            assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_completion_with_response_format(self, mock_config, mock_openai_response):
        """Test completion with JSON response format."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            messages = [{"role": "user", "content": "Hello"}]
            response_format = {"type": "json_object"}

            result = await client.chat_completion(
                messages,
                agent_id="test-agent",
                response_format=response_format
            )

            assert result.content == "Test response"
            # Verify response_format was passed
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["response_format"] == response_format

    @pytest.mark.asyncio
    async def test_token_tracking(self, mock_config, mock_openai_response):
        """Test token usage tracking per agent."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            messages = [{"role": "user", "content": "Hello"}]

            # First completion
            await client.chat_completion(messages, agent_id="agent1")
            usage1 = client.get_token_usage("agent1")
            assert usage1.prompt_tokens == 10
            assert usage1.completion_tokens == 20
            assert usage1.total_tokens == 30

            # Second completion for same agent
            await client.chat_completion(messages, agent_id="agent1")
            usage1 = client.get_token_usage("agent1")
            assert usage1.prompt_tokens == 20
            assert usage1.completion_tokens == 40
            assert usage1.total_tokens == 60

            # Completion for different agent
            await client.chat_completion(messages, agent_id="agent2")
            usage2 = client.get_token_usage("agent2")
            assert usage2.prompt_tokens == 10
            assert usage2.completion_tokens == 20
            assert usage2.total_tokens == 30

    @pytest.mark.asyncio
    async def test_reset_token_usage(self, mock_config, mock_openai_response):
        """Test resetting token usage."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            messages = [{"role": "user", "content": "Hello"}]

            # Track usage for two agents
            await client.chat_completion(messages, agent_id="agent1")
            await client.chat_completion(messages, agent_id="agent2")

            # Reset specific agent
            client.reset_token_usage("agent1")
            usage1 = client.get_token_usage("agent1")
            assert usage1.total_tokens == 0

            # agent2 should still have usage
            usage2 = client.get_token_usage("agent2")
            assert usage2.total_tokens == 30

            # Reset all
            client.reset_token_usage()
            usage2 = client.get_token_usage("agent2")
            assert usage2.total_tokens == 0


class TestConcurrency:
    """Test concurrent request handling."""

    @pytest.mark.asyncio
    async def test_batch_completions(self, mock_config, mock_openai_response):
        """Test batch completion processing."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.return_value = mock_openai_response

            requests = [
                {
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "agent_id": f"agent{i}"
                }
                for i in range(5)
            ]

            results = await client.batch_completions(requests)

            assert len(results) == 5
            assert all(isinstance(r, CompletionResult) for r in results)
            assert all(r.content == "Test response" for r in results)
            assert mock_create.call_count == 5

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, mock_config):
        """Test semaphore enforces concurrency limit."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(3)  # Limit to 3 concurrent

        concurrent_count = 0
        max_concurrent = 0

        async def mock_create_with_tracking(**kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate work
            concurrent_count -= 1

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
            mock_response.usage = MagicMock(
                prompt_tokens=5, completion_tokens=10, total_tokens=15
            )
            return mock_response

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = mock_create_with_tracking

            # Launch 10 concurrent requests
            tasks = [
                client.chat_completion(
                    [{"role": "user", "content": f"Request {i}"}],
                    agent_id=f"agent{i}"
                )
                for i in range(10)
            ]

            await asyncio.gather(*tasks)

            # Verify max concurrent was limited by semaphore
            assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_semaphore_limits_sequential(self, mock_config):
        """Test semaphore enforces sequential execution when limited to 1."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(1)  # Sequential

        concurrent_count = 0
        max_concurrent = 0

        async def mock_create_with_tracking(**kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="OK"))]
            mock_response.usage = MagicMock(
                prompt_tokens=5, completion_tokens=10, total_tokens=15
            )
            return mock_response

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = mock_create_with_tracking

            tasks = [
                client.chat_completion(
                    [{"role": "user", "content": f"Request {i}"}],
                    agent_id=f"agent{i}"
                )
                for i in range(5)
            ]

            await asyncio.gather(*tasks)

            # Verify strictly sequential execution
            assert max_concurrent == 1


class TestRetry:
    """Test retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_config):
        """Test retry mechanism on API failures."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        attempt_count = 0

        async def mock_create_with_retry(**kwargs):
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count < 3:
                raise Exception("Temporary failure")

            # Success on third attempt
            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Success"))]
            mock_response.usage = MagicMock(
                prompt_tokens=5, completion_tokens=10, total_tokens=15
            )
            return mock_response

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = mock_create_with_retry

            result = await client.chat_completion(
                [{"role": "user", "content": "Test"}],
                agent_id="test"
            )

            assert result.content == "Success"
            assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, mock_config):
        """Test behavior when all retry attempts fail."""
        client = InferenceClient(mock_config)
        client.backend = Backend.LLAMA_CPP
        client._semaphore = asyncio.Semaphore(8)

        with patch.object(
            client.client.chat.completions,
            "create",
            new_callable=AsyncMock
        ) as mock_create:
            mock_create.side_effect = Exception("Persistent failure")

            with pytest.raises(Exception, match="Persistent failure"):
                await client.chat_completion(
                    [{"role": "user", "content": "Test"}],
                    agent_id="test"
                )

            # Should have attempted 3 times
            assert mock_create.call_count == 3


class TestHealthCheck:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_returns_models(self, mock_config):
        """Test health check returns available models."""
        client = InferenceClient(mock_config)

        mock_response = {
            "data": [
                {"id": "model1", "root": "meta"},
                {"id": "model2", "root": "mistral"}
            ]
        }

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_httpx.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=MagicMock(
                    json=lambda: mock_response,
                    raise_for_status=lambda: None
                )
            )

            result = await client.health_check()

            assert result["status"] == "healthy"
            assert result["models"] == ["model1", "model2"]
            assert result["max_concurrent"] == 8

    @pytest.mark.asyncio
    async def test_auto_health_check_on_completion(self, mock_config, mock_openai_response):
        """Test health check runs automatically before first completion."""
        client = InferenceClient(mock_config)

        assert client.backend is None
        assert client._semaphore is None

        # Mock health check
        with patch.object(client, "health_check", new_callable=AsyncMock) as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "backend": Backend.LLAMA_CPP,
                "models": ["test-model"]
            }

            # Set backend and semaphore as health_check would
            async def setup_client():
                client.backend = Backend.LLAMA_CPP
                client._semaphore = asyncio.Semaphore(8)
                return {"status": "healthy"}

            mock_health.side_effect = setup_client

            with patch.object(
                client.client.chat.completions,
                "create",
                new_callable=AsyncMock
            ) as mock_create:
                mock_create.return_value = mock_openai_response

                await client.chat_completion(
                    [{"role": "user", "content": "Test"}],
                    agent_id="test"
                )

                # Verify health check was called
                mock_health.assert_called_once()


class TestSingletonPattern:
    """Test singleton client pattern."""

    @pytest.mark.asyncio
    async def test_get_client_singleton(self, mock_config):
        """Test get_client returns same instance."""
        await reset_client()

        async def mock_health(self_inner):
            self_inner.backend = Backend.LLAMA_CPP
            if self_inner._max_concurrent is None:
                self_inner._max_concurrent = 8
            self_inner._semaphore = asyncio.Semaphore(self_inner._max_concurrent)
            return {"status": "healthy", "backend": self_inner.backend, "models": ["model1"]}

        with patch.object(InferenceClient, "health_check", mock_health):
            client1 = await get_client(mock_config)
            client2 = await get_client(mock_config)

            assert client1 is client2

        await reset_client()

    @pytest.mark.asyncio
    async def test_reset_client(self, mock_config):
        """Test reset_client clears singleton."""
        await reset_client()

        async def mock_health(self_inner):
            self_inner.backend = Backend.LLAMA_CPP
            if self_inner._max_concurrent is None:
                self_inner._max_concurrent = 8
            self_inner._semaphore = asyncio.Semaphore(self_inner._max_concurrent)
            return {"status": "healthy", "backend": self_inner.backend, "models": ["model1"]}

        with patch.object(InferenceClient, "health_check", mock_health):
            client1 = await get_client(mock_config)
            await reset_client()
            client2 = await get_client(mock_config)

            assert client1 is not client2

        await reset_client()
