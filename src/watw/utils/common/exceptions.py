"""
Common exceptions for the WATW project.

This module provides common exception classes used throughout the project.
"""

from enum import Enum, auto
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    """Status of an API task."""

    PENDING = auto()
    PROCESSING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class WATWError(Exception):
    """Base exception class for all WATW errors."""

    pass


class ConfigurationError(WATWError):
    """Exception raised for configuration errors."""

    pass


class ValidationError(WATWError):
    """Exception raised for validation errors."""

    pass


class FileOperationError(WATWError):
    """Exception raised for file operation errors."""

    pass


class APIError(WATWError):
    """Base exception class for API errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class APITimeoutError(APIError):
    """Exception raised when an API request times out."""

    def __init__(self, service: str, timeout: int):
        super().__init__(f"{service} API request timed out after {timeout} seconds")
        self.service = service
        self.timeout = timeout


class APIResponseError(APIError):
    """Exception raised when an API response is invalid."""

    pass


class TensorArtError(APIError):
    """Exception raised for TensorArt API errors."""

    pass


class RunwayMLError(APIError):
    """Exception raised for RunwayML API errors."""

    pass


class ElevenLabsError(APIError):
    """Exception raised for ElevenLabs API errors."""

    pass


class WorkflowError(WATWError):
    """Raised when there are issues with the overall workflow execution."""

    pass


class FFmpegError(WATWError):
    """Raised when FFmpeg operations fail."""

    def __init__(
        self,
        message: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
    ):
        super().__init__(message)
        self.command = command
        self.output = output


class RateLimitExceeded(Exception):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")
