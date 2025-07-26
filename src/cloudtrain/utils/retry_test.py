"""Unit tests for CloudTrain retry utilities."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from cloudtrain.utils.retry import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    RetryError,
    circuit_breaker,
    retry,
    retry_with_backoff,
)


class TestRetryWithBackoff:
    """Test retry_with_backoff function."""

    @pytest.mark.asyncio
    async def test_successful_function_no_retry(self):
        """Test successful function execution without retries."""

        async def success_func():
            return "success"

        result = await retry_with_backoff(success_func, max_retries=3)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_successful_function_with_args(self):
        """Test successful function execution with arguments."""

        async def add_func(a, b, multiplier=1):
            return (a + b) * multiplier

        result = await retry_with_backoff(add_func, 2, 3, max_retries=3, multiplier=2)
        assert result == 10

    @pytest.mark.asyncio
    async def test_function_succeeds_after_retries(self):
        """Test function that succeeds after some failures."""
        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = await retry_with_backoff(
            flaky_func, max_retries=3, base_delay=0.01  # Fast for testing
        )
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_function_fails_all_retries(self):
        """Test function that fails all retry attempts."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(RetryError) as exc_info:
            await retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

        assert "Function failed after 3 attempts" in str(exc_info.value)
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert call_count == 3  # Initial + 2 retries

    @pytest.mark.asyncio
    async def test_sync_function_support(self):
        """Test retry with synchronous functions."""
        call_count = 0

        def sync_flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "sync_success"

        result = await retry_with_backoff(
            sync_flaky_func, max_retries=2, base_delay=0.01
        )
        assert result == "sync_success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_specific_exceptions(self):
        """Test retry only on specific exception types."""
        call_count = 0

        async def specific_error_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Retryable error")
            elif call_count == 2:
                raise TypeError("Non-retryable error")
            return "success"

        # Should not retry on TypeError
        with pytest.raises(TypeError):
            await retry_with_backoff(
                specific_error_func, max_retries=3, base_delay=0.01, retry_on=ValueError
            )

        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self):
        """Test exponential backoff timing."""
        import time

        call_times = []

        async def timing_func():
            call_times.append(time.time())
            if len(call_times) < 3:
                raise ValueError("Temporary failure")
            return "success"

        start_time = time.time()
        await retry_with_backoff(
            timing_func,
            max_retries=3,
            base_delay=0.1,
            exponential_base=2.0,
            jitter=False,  # Disable jitter for predictable timing
        )

        # Verify exponential backoff
        assert len(call_times) == 3

        # First retry should be after ~0.1 seconds
        delay1 = call_times[1] - call_times[0]
        assert 0.08 <= delay1 <= 0.15

        # Second retry should be after ~0.2 seconds
        delay2 = call_times[2] - call_times[1]
        assert 0.18 <= delay2 <= 0.25


class TestRetryDecorator:
    """Test retry decorator."""

    @pytest.mark.asyncio
    async def test_async_function_decorator(self):
        """Test retry decorator with async function."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.01)
        async def decorated_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "decorated_success"

        result = await decorated_async_func()
        assert result == "decorated_success"
        assert call_count == 2

    def test_sync_function_decorator(self):
        """Test retry decorator with sync function."""
        call_count = 0

        @retry(max_retries=2, base_delay=0.01)
        def decorated_sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary failure")
            return "sync_decorated_success"

        result = decorated_sync_func()
        assert result == "sync_decorated_success"
        assert call_count == 2


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        breaker = CircuitBreaker(
            failure_threshold=3, recovery_timeout=30.0, expected_exception=ValueError
        )

        assert breaker.failure_threshold == 3
        assert breaker.recovery_timeout == 30.0
        assert breaker.expected_exception == ValueError
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_success(self):
        """Test circuit breaker with successful function."""
        breaker = CircuitBreaker(failure_threshold=2)

        async def success_func():
            return "success"

        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)

        async def failing_func():
            raise ValueError("Always fails")

        # First two failures should work but increment failure count
        for i in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_func)
            assert breaker.state == "CLOSED"

        # Third failure should open the circuit
        with pytest.raises(ValueError):
            await breaker.call(failing_func)
        assert breaker.state == "OPEN"
        assert breaker.failure_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=60.0)

        async def failing_func():
            raise ValueError("Failure")

        # Trigger circuit to open
        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        assert breaker.state == "OPEN"

        # Now calls should be blocked
        async def any_func():
            return "should not execute"

        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(any_func)

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        async def failing_func():
            raise ValueError("Failure")

        async def success_func():
            return "success"

        # Open the circuit
        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        assert breaker.state == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Next call should enter half-open state and succeed
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_ignores_unexpected_exceptions(self):
        """Test circuit breaker ignores unexpected exception types."""
        breaker = CircuitBreaker(failure_threshold=2, expected_exception=ValueError)

        async def type_error_func():
            raise TypeError("Different error type")

        # TypeError should not trigger circuit breaker
        with pytest.raises(TypeError):
            await breaker.call(type_error_func)

        assert breaker.state == "CLOSED"
        assert breaker.failure_count == 0


class TestCircuitBreakerDecorator:
    """Test circuit breaker decorator."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_decorator(self):
        """Test circuit breaker decorator functionality."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, expected_exception=ValueError)
        async def decorated_func(should_fail=True):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise ValueError("Decorated failure")
            return f"success_{call_count}"

        # First two failures
        for i in range(2):
            with pytest.raises(ValueError):
                await decorated_func(should_fail=True)

        # Third failure should open circuit
        with pytest.raises(ValueError):
            await decorated_func(should_fail=True)

        # Now circuit should be open
        with pytest.raises(CircuitBreakerOpenError):
            await decorated_func(should_fail=False)  # Even success calls are blocked

        assert call_count == 3  # Only the first 3 calls executed


class TestRetryIntegration:
    """Test integration between retry and circuit breaker."""

    @pytest.mark.asyncio
    async def test_retry_with_circuit_breaker(self):
        """Test retry mechanism with circuit breaker protection."""
        call_count = 0
        breaker = CircuitBreaker(failure_threshold=3)

        async def protected_flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("Temporary failure")
            return "protected_success"

        # Use retry with circuit breaker
        async def retry_with_breaker():
            return await breaker.call(protected_flaky_func)

        result = await retry_with_backoff(
            retry_with_breaker, max_retries=3, base_delay=0.01
        )

        assert result == "protected_success"
        assert call_count == 3
        assert breaker.state == "CLOSED"  # Should reset after success
