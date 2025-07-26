"""Retry utilities for CloudTrain.

This module provides retry logic and backoff strategies for handling
transient failures when interacting with cloud provider APIs.
"""

import asyncio
import logging
import random
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


class RetryError(Exception):
    """Exception raised when retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None):
        super().__init__(message)
        self.last_exception = last_exception


async def retry_with_backoff(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
    **kwargs,
) -> Any:
    """Retry a function with exponential backoff.

    This function implements a robust retry mechanism with exponential backoff
    and optional jitter to handle transient failures in cloud API calls.

    Args:
        func: Function to retry (can be sync or async)
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for the first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retry_on: Exception types to retry on (None means retry on all)
        **kwargs: Keyword arguments to pass to the function

    Returns:
        Result of the function call

    Raises:
        RetryError: If all retry attempts are exhausted
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            # Call the function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            if attempt > 0:
                logger.info(f"Function succeeded on attempt {attempt + 1}")

            return result

        except Exception as e:
            last_exception = e

            # Check if we should retry on this exception
            if retry_on is not None:
                if not isinstance(e, retry_on):
                    logger.debug(f"Not retrying on exception type: {type(e).__name__}")
                    raise

            # Don't retry on the last attempt
            if attempt == max_retries:
                logger.error(f"Function failed after {max_retries + 1} attempts")
                break

            # Calculate delay for next attempt
            delay = min(base_delay * (exponential_base**attempt), max_delay)

            # Add jitter if enabled
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)

            logger.warning(
                f"Function failed on attempt {attempt + 1}/{max_retries + 1}: {e}. "
                f"Retrying in {delay:.2f} seconds..."
            )

            await asyncio.sleep(delay)

    # All retries exhausted
    raise RetryError(
        f"Function failed after {max_retries + 1} attempts", last_exception
    )


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retry_on: Optional[Union[Type[Exception], Tuple[Type[Exception], ...]]] = None,
):
    """Decorator for adding retry logic to functions.

    This decorator adds retry functionality to both synchronous and
    asynchronous functions with configurable backoff strategies.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for the first retry
        max_delay: Maximum delay in seconds between retries
        exponential_base: Base for exponential backoff calculation
        jitter: Whether to add random jitter to delays
        retry_on: Exception types to retry on (None means retry on all)

    Returns:
        Decorated function with retry logic

    Example:
        @retry(max_retries=3, base_delay=1.0)
        async def api_call():
            # This function will be retried up to 3 times
            response = await some_api_call()
            return response
    """

    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_with_backoff(
                    func,
                    *args,
                    max_retries=max_retries,
                    base_delay=base_delay,
                    max_delay=max_delay,
                    exponential_base=exponential_base,
                    jitter=jitter,
                    retry_on=retry_on,
                    **kwargs,
                )

            return async_wrapper
        else:

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, we need to run in an event loop
                return asyncio.run(
                    retry_with_backoff(
                        func,
                        *args,
                        max_retries=max_retries,
                        base_delay=base_delay,
                        max_delay=max_delay,
                        exponential_base=exponential_base,
                        jitter=jitter,
                        retry_on=retry_on,
                        **kwargs,
                    )
                )

            return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance.

    This class implements the circuit breaker pattern to prevent
    cascading failures by temporarily stopping calls to a failing service.

    Attributes:
        failure_threshold: Number of failures before opening the circuit
        recovery_timeout: Time to wait before attempting recovery
        expected_exception: Exception type that triggers the circuit breaker
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type that triggers the breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call a function through the circuit breaker.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function call

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from the function
        """
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
                logger.info("Circuit breaker entering HALF_OPEN state")
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            # Call the function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset failure count
            self._on_success()
            return result

        except self.expected_exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.last_failure_time is None:
            return True

        import time

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful function call."""
        self.failure_count = 0
        self.state = "CLOSED"
        if self.last_failure_time is not None:
            logger.info("Circuit breaker reset to CLOSED state")
            self.last_failure_time = None

    def _on_failure(self) -> None:
        """Handle failed function call."""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count > self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
):
    """Decorator for adding circuit breaker functionality.

    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Seconds to wait before attempting recovery
        expected_exception: Exception type that triggers the breaker

    Returns:
        Decorated function with circuit breaker logic
    """
    breaker = CircuitBreaker(failure_threshold, recovery_timeout, expected_exception)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator
