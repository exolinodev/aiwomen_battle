"""
Rate limiting and retry utilities for API calls.

This module provides utilities for handling API rate limits and implementing
retry logic for API calls. It includes exponential backoff strategies and
configurable retry parameters.
"""

import time
import logging
import random
from typing import Callable, Any, Dict, Optional, TypeVar, Type, Union, List
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic return type
T = TypeVar('T')

class RateLimitExceeded(Exception):
    """Exception raised when a rate limit is exceeded."""
    def __init__(self, message: str, retry_after: Optional[int] = None):
        self.message = message
        self.retry_after = retry_after
        super().__init__(self.message)

class RetryableError(Exception):
    """Base exception for errors that can be retried."""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_data: Optional[Dict[str, Any]] = None):
        self.message = message
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(self.message)

class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        exponential_backoff: bool = True,
        retry_on_exceptions: Optional[List[Type[Exception]]] = None,
        retry_on_status_codes: Optional[List[int]] = None
    ):
        """
        Initialize retry configuration.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            jitter: Whether to add random jitter to delays
            exponential_backoff: Whether to use exponential backoff
            retry_on_exceptions: List of exception types to retry on
            retry_on_status_codes: List of HTTP status codes to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.exponential_backoff = exponential_backoff
        self.retry_on_exceptions = retry_on_exceptions or [
            RetryableError,
            RateLimitExceeded,
            ConnectionError,
            TimeoutError
        ]
        self.retry_on_status_codes = retry_on_status_codes or [
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504   # Gateway Timeout
        ]

def calculate_delay(retry_count: int, config: RetryConfig) -> float:
    """
    Calculate delay for next retry using exponential backoff.
    
    Args:
        retry_count: Current retry count (0-based)
        config: Retry configuration
        
    Returns:
        float: Delay in seconds before next retry
    """
    if not config.exponential_backoff:
        delay = config.base_delay
    else:
        # Exponential backoff: base_delay * 2^retry_count
        delay = min(config.base_delay * (2 ** retry_count), config.max_delay)
    
    # Add jitter if enabled
    if config.jitter:
        # Add random jitter between 0% and 25% of the delay
        jitter_factor = random.uniform(0, 0.25)
        delay = delay * (1 + jitter_factor)
    
    return delay

def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
) -> Callable:
    """
    Decorator for adding retry logic to a function.
    
    Args:
        config: Retry configuration
        on_retry: Callback function called before each retry
        
    Returns:
        Callable: Decorated function with retry logic
    """
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
                    should_retry = False
                    
                    # Check if it's a rate limit exception
                    if isinstance(e, RateLimitExceeded):
                        should_retry = True
                        retry_after = e.retry_after
                        if retry_after:
                            logger.info(f"Rate limit exceeded. Retry after {retry_after} seconds.")
                            time.sleep(retry_after)
                            continue
                    
                    # Check if it's a retryable exception
                    if any(isinstance(e, exc_type) for exc_type in config.retry_on_exceptions):
                        should_retry = True
                    
                    # Check if it's a requests exception with a retryable status code
                    if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                        status_code = e.response.status_code
                        if status_code in config.retry_on_status_codes:
                            should_retry = True
                            logger.info(f"Received status code {status_code}, will retry.")
                    
                    # If we shouldn't retry or we're out of retries, re-raise
                    if not should_retry or retry_count >= config.max_retries:
                        raise
                    
                    # Calculate delay for next retry
                    delay = calculate_delay(retry_count, config)
                    
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

def extract_retry_after(response: Any) -> Optional[int]:
    """
    Extract Retry-After header from response.
    
    Args:
        response: Response object
        
    Returns:
        Optional[int]: Retry-After value in seconds, or None if not found
    """
    if hasattr(response, 'headers'):
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            try:
                return int(retry_after)
            except ValueError:
                # If Retry-After is a date, return None
                return None
    return None

def handle_rate_limit(
    response: Any,
    max_retries: int = 5,
    base_delay: float = 60.0
) -> None:
    """
    Handle rate limit response.
    
    Args:
        response: Response object
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        
    Raises:
        RateLimitExceeded: If rate limit is exceeded
    """
    if hasattr(response, 'status_code') and response.status_code == 429:
        retry_after = extract_retry_after(response)
        if retry_after:
            raise RateLimitExceeded("Rate limit exceeded", retry_after)
        else:
            # If no Retry-After header, use exponential backoff
            raise RateLimitExceeded("Rate limit exceeded") 