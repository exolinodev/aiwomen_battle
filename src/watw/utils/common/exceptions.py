"""
Custom exceptions for the Women Around The World project.
"""

class WATWError(Exception):
    """Base exception class for all WATW-related errors."""
    pass

class ValidationError(WATWError):
    """Raised when validation fails for any input or configuration."""
    pass

class APIError(WATWError):
    """Base class for API-related errors."""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data

class TensorArtError(APIError):
    """Raised when TensorArt API operations fail."""
    pass

class RunwayMLError(APIError):
    """Raised when RunwayML API operations fail."""
    pass

class FileOperationError(WATWError):
    """Raised when file operations (read/write/download) fail."""
    pass

class ConfigurationError(WATWError):
    """Raised when there are issues with configuration or credentials."""
    pass

class WorkflowError(WATWError):
    """Raised when there are issues with the overall workflow execution."""
    pass

class FFmpegError(WATWError):
    """Raised when FFmpeg operations fail."""
    def __init__(self, message: str, command: str = None, output: str = None):
        super().__init__(message)
        self.command = command
        self.output = output

class RateLimitExceeded(Exception):
    """Exception raised when API rate limit is exceeded."""
    def __init__(self, retry_after: int = 60):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds") 