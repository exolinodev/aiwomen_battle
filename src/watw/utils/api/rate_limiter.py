"""
Rate limiting utilities for API clients.

This module provides classes and utilities for implementing
rate limiting for API clients, including token bucket and
fixed window rate limiters.
"""

import logging
import threading
import time
from functools import wraps
from typing import Any, Dict, Optional, TypeVar, cast, List, Union, Callable
from typing_extensions import ParamSpec

from watw.utils.api.retry import RateLimitExceeded

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for the decorated function
P = ParamSpec("P")
T = TypeVar("T")


class RateLimiter:
    """Base class for rate limiters."""

    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize the rate limiter.

        Args:
            service_name: Optional name of the service being rate limited
        """
        self.service_name = service_name or "api_service"
        self.logger = logging.getLogger(f"{__name__}.{self.service_name}")

    def acquire(self) -> bool:
        """Acquire a token from the rate limiter.

        Returns:
            bool: True if a token was acquired, False otherwise.
        """
        raise NotImplementedError()

    def wait_until_allowed(self) -> None:
        """Wait until a token can be acquired."""
        while not self.acquire():
            time.sleep(0.1)


class TokenBucketRateLimiter(RateLimiter):
    """
    Token bucket rate limiter.

    This implementation uses a token bucket algorithm where tokens are added
    at a constant rate up to a maximum capacity. Each request consumes one token.
    """

    def __init__(self, rate: float, burst: int, service_name: Optional[str] = None):
        """
        Initialize the token bucket rate limiter.

        Args:
            rate: Tokens per second
            burst: Maximum number of tokens that can be accumulated
            service_name: Optional name of the service being rate limited
        """
        super().__init__(service_name)

        self.rate = float(rate)
        self.burst = int(burst)
        self.tokens = float(burst)
        self.last_update = time.time()
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """Acquire a token from the rate limiter.

        Returns:
            bool: True if a token was acquired, False otherwise.
        """
        with self.lock:
            now = time.time()
            time_passed = now - self.last_update
            self.tokens = min(float(self.burst), self.tokens + time_passed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                self.logger.debug(
                    f"Token acquired. Remaining tokens: {self.tokens:.2f}"
                )
                return True
            else:
                self.logger.debug("Token not available")
                return False

    def wait_until_allowed(self) -> None:
        """Wait until a token can be acquired."""
        while not self.acquire():
            time.sleep(0.1)


class FixedWindowRateLimiter(RateLimiter):
    """
    Fixed window rate limiter.

    This implementation uses a fixed time window approach where a
    certain number of requests are allowed within each time window.
    """

    def __init__(
        self,
        requests_per_window: int,
        window_size: float,
        service_name: Optional[str] = None,
    ):
        """
        Initialize the fixed window rate limiter.

        Args:
            requests_per_window: Number of requests allowed per window
            window_size: Window size in seconds
            service_name: Optional name of the service being rate limited
        """
        super().__init__(service_name)

        self.requests_per_window = int(requests_per_window)  # Ensure int
        self.window_size = float(window_size)  # Ensure float
        self.window_start = time.time()
        self.request_count = 0
        self.lock = threading.RLock()

    def _check_window(self) -> None:
        """Check if we need to start a new window."""
        now = time.time()
        elapsed = now - self.window_start

        if elapsed >= self.window_size:
            # Start new window
            self.window_start = now
            self.request_count = 0

    def acquire(self) -> bool:
        """
        Acquire permission to make a request if within rate limit.

        Returns:
            bool: True if request is allowed, False otherwise
        """
        with self.lock:
            self._check_window()

            if self.request_count < self.requests_per_window:
                self.request_count += 1
                self.logger.debug(
                    f"Request allowed. Count: {self.request_count}/{self.requests_per_window}"
                )
                return True
            else:
                self.logger.debug(
                    f"Rate limit reached: {self.request_count}/{self.requests_per_window}"
                )
                return False


class RateLimiterRegistry:
    """Registry for rate limiters."""

    def __init__(self) -> None:
        """Initialize the rate limiter registry."""
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.lock = threading.Lock()

    def register_rate_limiter(self, service: str, rate_limiter: RateLimiter) -> None:
        """Register a rate limiter for a service.

        Args:
            service: Service name
            rate_limiter: Rate limiter instance
        """
        with self.lock:
            self.rate_limiters[service] = rate_limiter

    def get_rate_limiter(self, service: str) -> Optional[RateLimiter]:
        """Get the rate limiter for a service.

        Args:
            service: Service name

        Returns:
            Optional[RateLimiter]: Rate limiter instance, or None if not found
        """
        with self.lock:
            return self.rate_limiters.get(service)


def rate_limited(service: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to rate limit function calls.

    Args:
        service: Service name to rate limit

    Returns:
        Callable: Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            rate_limiter = service_rate_limiter.get_rate_limiter(service)
            if rate_limiter is None:
                logger.warning(f"No rate limiter found for service {service}")
                return func(*args, **kwargs)

            rate_limiter.wait_until_allowed()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def configure_rate_limiter(service: str, rate: float, burst: int) -> None:
    """Configure a rate limiter for a service.

    Args:
        service: Service name
        rate: Tokens per second
        burst: Maximum number of tokens that can be accumulated
    """
    rate_limiter = TokenBucketRateLimiter(rate, burst)
    service_rate_limiter.register_rate_limiter(service, rate_limiter)


# Global registry instance
service_rate_limiter = RateLimiterRegistry()
