"""
Backoff utilities for API clients.

This module provides utilities for implementing backoff strategies for
API clients, including exponential backoff with jitter.
"""

import logging
import random
import time
from typing import Any, Callable, Optional, TypeVar
from typing_extensions import ParamSpec

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for the decorated function
P = ParamSpec("P")
T = TypeVar("T")


def calculate_backoff(
    attempt: int, base_delay: float = 1.0, max_delay: float = 60.0, jitter: bool = True
) -> float:
    """
    Calculate backoff time using exponential backoff with jitter.

    Args:
        attempt: Current attempt number (0-based)
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add jitter to the delay

    Returns:
        float: Delay time in seconds
    """
    # Calculate exponential backoff
    delay = min(max_delay, base_delay * (2**attempt))

    # Add jitter if requested
    if jitter:
        jitter_amount = random.random() * 0.1 * delay
        delay = delay + jitter_amount

    return float(delay)


def exponential_backoff(
    func: Callable[P, T],
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None,
) -> Callable[P, T]:
    """
    Decorator for implementing exponential backoff with jitter.

    Args:
        func: Function to decorate
        max_attempts: Maximum number of attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        jitter: Whether to add jitter to the delay
        on_retry: Optional callback function to call before each retry

    Returns:
        Decorated function with exponential backoff
    """

    def wrapper(*args: Any, **kwargs: Any) -> T:
        attempt = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    logger.error(
                        f"Max attempts ({max_attempts}) exceeded. Last error: {e}"
                    )
                    raise

                delay = calculate_backoff(attempt, base_delay, max_delay, jitter)
                if on_retry:
                    on_retry(e, attempt, delay)

                logger.warning(
                    f"Attempt {attempt}/{max_attempts} failed. "
                    f"Retrying in {delay:.2f} seconds. Error: {e}"
                )
                time.sleep(delay)

    return wrapper
