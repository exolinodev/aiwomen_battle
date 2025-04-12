"""
Exceptions for API-related errors.
"""

from enum import Enum, auto


class TaskStatus(Enum):
    """Status of an API task."""

    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class APIError(Exception):
    """Base exception for API-related errors."""

    pass


class APIResponseError(APIError):
    """Exception raised when an API response indicates an error."""

    pass


class APITimeoutError(APIError):
    """Exception raised when an API request times out."""

    pass 