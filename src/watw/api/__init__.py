"""
API clients for Women Around The World project.

This package provides API clients for various services used in the project:
- ElevenLabs for text-to-speech
- Runway for image-to-video
- TensorArt for image generation
"""

from enum import Enum, auto

from . import config
from .clients.tensorart import TensorArtClient
from watw.utils.common.exceptions import APIError, APIResponseError, APITimeoutError, TaskStatus

__all__ = [
    "config",
    "TensorArtClient",
    "APIError",
    "APIResponseError",
    "APITimeoutError",
    "TaskStatus",
]


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
