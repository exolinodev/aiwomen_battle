"""
Retry utilities for API clients.

This module provides utilities for implementing retry logic for API clients,
including functions for retrying requests with exponential backoff.
"""

import functools
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Type, TypeVar, Union
from typing_extensions import ParamSpec

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for the decorated function
P = ParamSpec("P")
T = TypeVar("T")


class RetryableError(Exception):
    """Base class for errors that can be retried."""

    pass


class RateLimitExceeded(RetryableError):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, retry_after: Optional[Union[int, float]] = None):
        """
        Initialize the RateLimitExceeded exception.

        Args:
            message: Error message
            retry_after: Time to wait before retrying (in seconds)
        """
        self.message = message
        self.retry_after = retry_after
        super().__init__(
            f"{message}. Retry after: {retry_after if retry_after is not None else 'Unknown'} seconds"
        )


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    """Maximum number of retry attempts."""

    base_delay: float = 1.0
    """Base delay between retries in seconds."""

    max_delay: float = 60.0
    """Maximum delay between retries in seconds."""

    retry_on: List[Union[int, Type[Exception]]] = field(
        default_factory=lambda: [429, 500, 502, 503, 504]
    )
    """List of status codes or exception types to retry on."""

    retry_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "PATCH"]
    )
    """List of HTTP methods to retry."""

    jitter: bool = True
    """Whether to apply jitter to the delay."""

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields."""
        if self.retry_on is None:
            self.retry_on = [429, 500, 502, 503, 504]
        if self.retry_methods is None:
            self.retry_methods = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        return None


def extract_retry_after(response: Any) -> Optional[float]:
    """
    Extract the Retry-After header from a response.

    Args:
        response: Response object

    Returns:
        Optional[float]: Retry-After value in seconds, or None if not found
    """
    if not hasattr(response, "headers"):
        return None

    retry_after = response.headers.get("Retry-After")

    if retry_after is None:
        return None

    try:
        # Retry-After can be a timestamp or a number of seconds
        if retry_after.isdigit():
            return float(retry_after)
        else:
            # Try to parse as HTTP date
            from datetime import datetime
            from email.utils import parsedate_to_datetime

            retry_date = parsedate_to_datetime(retry_after)
            retry_seconds = (retry_date - datetime.now()).total_seconds()
            return retry_seconds
    except (ValueError, ImportError, AttributeError):
        # If we can't parse it, return None
        return None


def calculate_backoff(
    retry_attempt: int, config: RetryConfig, retry_after: Optional[float] = None
) -> float:
    """
    Calculate the backoff time for a retry attempt.

    Args:
        retry_attempt: Current retry attempt (0-based)
        config: Retry configuration
        retry_after: Optional retry after time from response

    Returns:
        float: Time to wait before retrying (in seconds)
    """
    # If we have a retry_after value, use that
    if retry_after is not None:
        return float(retry_after)

    # Otherwise, use exponential backoff
    delay = min(config.max_delay, config.base_delay * (2**retry_attempt))

    # Add jitter if configured
    if config.jitter:
        jitter = random.random() * 0.1 * delay
        delay = delay + jitter

    return float(delay)


def with_retry(
    config: Optional[RetryConfig] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        config: Optional retry configuration

    Returns:
        Decorator function
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Override config if provided in kwargs
            retry_config = config or RetryConfig()

            # Initialize retry attempt counter
            retry_attempt = 0
            last_exception = None

            while retry_attempt <= retry_config.max_retries:
                try:
                    # Execute the function
                    return func(*args, **kwargs)

                except Exception as e:
                    last_exception = e
                    
                    # Check if we should retry
                    if retry_attempt >= retry_config.max_retries:
                        logger.error(
                            f"Max retries ({retry_config.max_retries}) exceeded. Last error: {e}"
                        )
                        raise

                    # Check if the error is retryable
                    if not any(
                        (
                            isinstance(e, retry_type)
                            if isinstance(retry_type, type)
                            else getattr(e, "status_code", None) == retry_type
                        )
                        for retry_type in retry_config.retry_on
                    ):
                        raise

                    # Calculate backoff time
                    retry_after = getattr(e, "retry_after", None)
                    delay = calculate_backoff(retry_attempt, retry_config, retry_after)

                    # Log retry attempt
                    logger.warning(
                        f"Retry attempt {retry_attempt + 1}/{retry_config.max_retries} after error: {e}. "
                        f"Waiting {delay:.2f} seconds before retrying."
                    )

                    # Wait before retrying
                    time.sleep(delay)
                    retry_attempt += 1

            # This should never be reached in practice, but mypy needs it
            raise RuntimeError("Unexpected end of retry loop")

        return wrapper

    return decorator
