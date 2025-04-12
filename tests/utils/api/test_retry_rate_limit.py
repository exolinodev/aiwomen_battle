"""
Test script for the rate limiting and retry logic.
This script is a standalone implementation that doesn't depend on the project.
"""

import logging
import random
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

import pytest

from watw.utils.api.retry_rate_limit import (
    RateLimiter,
    RateLimitExceeded,
    RetryableError,
    RetryConfig,
    ServiceRateLimiter,
    TokenBucketRateLimiter,
    add_jitter,
    calculate_backoff,
    configure_rate_limiter,
    constant_backoff,
    exponential_backoff,
    get_backoff_function,
    linear_backoff,
    rate_limited,
    should_retry_exception,
    sleep_with_backoff,
    with_retry,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_test")

# === Backoff Module === #


def constant_backoff(  # noqa: F811
    base_delay: float, retry_count: int, max_delay: Optional[float] = None
) -> float:
    """Implement constant backoff strategy."""
    return base_delay


def linear_backoff(  # noqa: F811
    base_delay: float, retry_count: int, max_delay: Optional[float] = None
) -> float:
    """Implement linear backoff strategy."""
    delay = base_delay * (retry_count + 1)
    if max_delay is not None:
        delay = min(delay, max_delay)
    return delay


def exponential_backoff(  # noqa: F811
    base_delay: float, retry_count: int, max_delay: Optional[float] = None
) -> float:
    """Implement exponential backoff strategy."""
    delay = base_delay * (2**retry_count)
    if max_delay is not None:
        delay = min(delay, max_delay)
    return delay


def add_jitter(delay: float, factor: float = 0.25) -> float:  # noqa: F811
    """Add random jitter to a delay value."""
    jitter = random.uniform(-factor, factor)
    return delay * (1 + jitter)


def get_backoff_function(  # noqa: F811
    strategy: str,
) -> Callable[[float, int, Optional[float]], float]:
    """Get a backoff function by name."""
    strategies = {
        "constant": constant_backoff,
        "linear": linear_backoff,
        "exponential": exponential_backoff,
    }
    return strategies[strategy]


def calculate_backoff(  # noqa: F811
    retry_count: int,
    base_delay: float = 1.0,
    max_delay: Optional[float] = None,
    strategy: str = "exponential",
    jitter: bool = True,
) -> float:
    """Calculate backoff delay."""
    backoff_func = get_backoff_function(strategy)
    delay = backoff_func(base_delay, retry_count, max_delay)
    if jitter:
        delay = add_jitter(delay)
    return delay


def sleep_with_backoff(  # noqa: F811
    retry_count: int,
    base_delay: float = 1.0,
    max_delay: Optional[float] = None,
    strategy: str = "exponential",
    jitter: bool = True,
) -> None:
    """Sleep with backoff."""
    delay = calculate_backoff(retry_count, base_delay, max_delay, strategy, jitter)
    time.sleep(delay)


# === Retry Module === #

# Type variable for generic return type
T = TypeVar("T")


class RetryableError(Exception):  # noqa: F811
    """Base exception for errors that can be retried."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)


class RateLimitExceeded(RetryableError):  # noqa: F811
    """Exception raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        status_code: Optional[int] = None,
    ):
        self.retry_after = retry_after
        super().__init__(message, status_code)


class RetryConfig:  # noqa: F811
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_strategy: str = "exponential",
        jitter: bool = True,
        retry_on_exceptions: Optional[List[Type[Exception]]] = None,
        retry_on_status_codes: Optional[List[int]] = None,
        respect_retry_after: bool = True,
    ):
        """Initialize retry configuration."""
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_strategy = backoff_strategy
        self.jitter = jitter
        self.respect_retry_after = respect_retry_after

        # Set default exceptions if none provided
        self.retry_on_exceptions = retry_on_exceptions or [
            RetryableError,
            RateLimitExceeded,
            ConnectionError,
            TimeoutError,
        ]

        # Set default status codes if none provided
        self.retry_on_status_codes = retry_on_status_codes or [
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        ]


def should_retry_exception(exception: Exception, config: RetryConfig) -> bool:  # noqa: F811
    """Determine if an exception should be retried."""
    # Check if it's a retryable exception
    if any(isinstance(exception, exc_type) for exc_type in config.retry_on_exceptions):
        return True

    # Check if it's a requests exception with a retryable status code
    if hasattr(exception, "response") and hasattr(exception.response, "status_code"):
        if exception.response.status_code in config.retry_on_status_codes:
            return True

    return False


def with_retry(  # noqa: F811
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable:
    """Decorator for adding retry logic to a function."""
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception = None

            for retry_count in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e

                    # Check if we should retry this exception
                    should_retry = should_retry_exception(e, config)

                    # If we shouldn't retry or we're out of retries, re-raise
                    if not should_retry or retry_count >= config.max_retries:
                        raise

                    # Check if it's a rate limit exception with a Retry-After header
                    retry_after = None
                    if (
                        isinstance(e, RateLimitExceeded)
                        and e.retry_after
                        and config.respect_retry_after
                    ):
                        retry_after = e.retry_after

                    # Use the Retry-After value if available, otherwise calculate backoff
                    if retry_after:
                        delay = retry_after
                        logger.info(
                            f"Rate limit exceeded. Retry after {delay} seconds."
                        )
                    else:
                        delay = calculate_backoff(
                            retry_count=retry_count,
                            base_delay=config.base_delay,
                            max_delay=config.max_delay,
                            strategy=config.backoff_strategy,
                            jitter=config.jitter,
                        )

                    # Log retry attempt
                    logger.warning(
                        f"Attempt {retry_count + 1}/{config.max_retries} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )

                    # Call on_retry callback if provided
                    if on_retry:
                        on_retry(e, retry_count, delay)

                    # Wait before retrying
                    time.sleep(delay)

            # This should never be reached due to the raise in the loop
            raise last_exception

        return wrapper

    return decorator


# === Rate Limiter Module === #


class RateLimiter:  # noqa: F811
    """Base class for rate limiters."""

    def __init__(self, service_name: Optional[str] = None):
        """Initialize the rate limiter."""
        self.service_name = service_name or "api_service"
        self.logger = logger

    def acquire(self) -> bool:
        """Acquire permission to make a request."""
        raise NotImplementedError("Subclasses must implement this method")

    def wait_until_allowed(self, timeout: Optional[float] = None) -> bool:
        """Wait until a request is allowed."""
        start_time = time.time()

        while True:
            if self.acquire():
                return True

            if timeout is not None and time.time() - start_time >= timeout:
                return False

            time.sleep(0.1)  # Short sleep to avoid CPU spinning


class TokenBucketRateLimiter(RateLimiter):  # noqa: F811
    """Token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int, service_name: Optional[str] = None):
        """Initialize the token bucket rate limiter."""
        super().__init__(service_name)

        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill_timestamp = time.time()
        self.lock = None  # No locking needed for testing

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill_timestamp

        # Calculate new tokens to add
        new_tokens = elapsed * self.rate

        # Update token count and timestamp
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill_timestamp = now

    def acquire(self) -> bool:
        """Acquire a token if available."""
        self._refill()

        if self.tokens >= 1:
            self.tokens -= 1
            self.logger.debug(f"Token acquired. Remaining tokens: {self.tokens:.2f}")
            return True
        else:
            self.logger.debug("Token not available")
            return False


class ServiceRateLimiter:  # noqa: F811
    """Rate limiter for multiple services."""

    def __init__(self):
        """Initialize the service rate limiter."""
        self.rate_limiters: Dict[str, RateLimiter] = {}

    def register_rate_limiter(
        self, service_name: str, rate_limiter: RateLimiter
    ) -> None:
        """Register a rate limiter for a service."""
        self.rate_limiters[service_name] = rate_limiter

    def get_rate_limiter(self, service_name: str) -> Optional[RateLimiter]:
        """Get the rate limiter for a service."""
        return self.rate_limiters.get(service_name)

    def acquire(self, service_name: str) -> bool:
        """Acquire permission to make a request to a service."""
        rate_limiter = self.get_rate_limiter(service_name)

        if rate_limiter:
            return rate_limiter.acquire()
        else:
            # No rate limiter for this service, so allow the request
            return True

    def wait_until_allowed(
        self, service_name: str, timeout: Optional[float] = None
    ) -> bool:
        """Wait until a request to a service is allowed."""
        rate_limiter = self.get_rate_limiter(service_name)

        if rate_limiter:
            return rate_limiter.wait_until_allowed(timeout)
        else:
            # No rate limiter for this service, so allow the request
            return True


# Global service rate limiter instance
service_rate_limiter = ServiceRateLimiter()


def rate_limited(  # noqa: F811
    service_name: str, raise_on_limit: bool = True, timeout: Optional[float] = None
) -> Callable:
    """Decorator for rate limiting a function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            rate_limiter = service_rate_limiter.get_rate_limiter(service_name)

            if not rate_limiter:
                # No rate limiter for this service, so just call the function
                return func(*args, **kwargs)

            # Try to acquire permission
            allowed = rate_limiter.wait_until_allowed(timeout)

            if allowed:
                return func(*args, **kwargs)
            elif raise_on_limit:
                # Calculate time until next request would be allowed
                if isinstance(rate_limiter, TokenBucketRateLimiter):
                    wait_time = int((1.0 - rate_limiter.tokens) / rate_limiter.rate) + 1
                else:
                    wait_time = 60  # Default wait time

                raise RateLimitExceeded(
                    message=f"Rate limit exceeded for service: {service_name}",
                    retry_after=wait_time,
                )
            else:
                # Just return None if we don't want to raise
                return None

        return wrapper

    return decorator


def configure_rate_limiter(  # noqa: F811
    service_name: str, requests_per_minute: int
) -> RateLimiter:
    """Configure and register a rate limiter for a service."""
    # Calculate parameters
    rate = requests_per_minute / 60.0  # Convert to requests per second
    capacity = max(
        1, int(requests_per_minute / 4)
    )  # Capacity is 1/4 of the per-minute limit

    # Create and register rate limiter
    rate_limiter = TokenBucketRateLimiter(rate, capacity, service_name)
    service_rate_limiter.register_rate_limiter(service_name, rate_limiter)

    logger.info(
        f"Configured rate limiter for {service_name}: {requests_per_minute} requests/minute"
    )

    return rate_limiter


# === Test Functions === #


# Test backoff strategies
def test_backoff():
    """Test different backoff strategies."""
    # Test constant backoff
    assert constant_backoff(1.0, 0) == 1.0
    assert constant_backoff(1.0, 5) == 1.0

    # Test linear backoff
    assert linear_backoff(1.0, 0) == 1.0
    assert linear_backoff(1.0, 1) == 2.0
    assert linear_backoff(1.0, 2) == 3.0
    assert linear_backoff(1.0, 2, max_delay=2.5) == 2.5

    # Test exponential backoff
    assert exponential_backoff(1.0, 0) == 1.0
    assert exponential_backoff(1.0, 1) == 2.0
    assert exponential_backoff(1.0, 2) == 4.0
    assert exponential_backoff(1.0, 2, max_delay=3.0) == 3.0

    # Test jitter
    delay = 1.0
    jittered = add_jitter(delay)
    assert 0.75 <= jittered <= 1.25  # 25% jitter range

    # Test backoff function selection
    assert get_backoff_function("constant")(1.0, 0) == 1.0
    assert get_backoff_function("linear")(1.0, 1) == 2.0
    assert get_backoff_function("exponential")(1.0, 2) == 4.0

    with pytest.raises(ValueError):
        get_backoff_function("invalid")


# Test retry logic
def test_retry():
    """Test retry functionality."""

    # Create a flaky function that fails a certain number of times
    @with_retry(config=RetryConfig(max_retries=3, base_delay=0.1, jitter=False))
    def flaky_function(fail_count=2):
        nonlocal call_count
        call_count += 1
        if call_count <= fail_count:
            raise RetryableError("Temporary failure")
        return "success"

    # Test successful retry
    call_count = 0
    result = flaky_function(fail_count=2)
    assert result == "success"
    assert call_count == 3  # Initial attempt + 2 retries

    # Test max retries exceeded
    call_count = 0
    with pytest.raises(RetryableError):
        flaky_function(fail_count=4)
    assert call_count == 4  # Initial attempt + 3 retries


# Test rate limiting
def test_rate_limiting():
    """Test rate limiting functionality."""
    # Test token bucket rate limiter
    limiter = TokenBucketRateLimiter(rate=2.0, capacity=2)
    assert limiter.acquire()  # First request
    assert limiter.acquire()  # Second request
    assert not limiter.acquire()  # Third request should be denied

    # Wait for tokens to refill
    time.sleep(0.6)  # 2 tokens per second, so 0.5s per token
    assert limiter.acquire()  # Should get one token back

    # Test rate limited decorator
    @rate_limited(service_name="test_service", raise_on_limit=False, timeout=0.0)
    def rate_limited_function():
        return "success"

    # Configure rate limiter for test service
    configure_rate_limiter("test_service", requests_per_minute=60)

    # First request should succeed
    assert rate_limited_function() == "success"

    # Test rate limited decorator with raising
    @rate_limited(service_name="test_service", raise_on_limit=True)
    def rate_limited_function_raising():
        return "success"

    # First request should succeed
    assert rate_limited_function_raising() == "success"

    # Test service rate limiter
    service_limiter = ServiceRateLimiter()
    token_bucket = TokenBucketRateLimiter(rate=1.0, capacity=1)
    service_limiter.register_rate_limiter("test_service", token_bucket)

    assert service_limiter.acquire("test_service")
    assert not service_limiter.acquire("test_service")

    # Test non-existent service
    assert service_limiter.acquire("non_existent_service")  # Should allow by default


def main():
    logger.info("Starting API utils test")

    # Run the tests
    test_backoff()
    test_retry()
    test_rate_limiting()

    logger.info("API utils test completed")


if __name__ == "__main__":
    main()
