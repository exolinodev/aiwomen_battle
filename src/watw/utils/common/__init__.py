"""
Common utility modules for Women Around The World.

This package provides common utility functions and classes used throughout
the Women Around The World application.
"""

from watw.api import config as api_config
from watw.utils.common.exceptions import (
    APIError,
    APIResponseError,
    APITimeoutError,
    TaskStatus,
)
from watw.utils.common.file_utils import (
    copy_file,
    create_temp_file,
    ensure_directory,
    get_file_extension,
)
from watw.utils.common.logging_utils import (
    log_execution_time,
    log_function_call,
    setup_logger,
)
from watw.utils.common.media_utils import (
    FFmpegError,
    MediaError,
    add_background_music,
    combine_video_with_audio,
    concatenate_videos,
    extract_audio,
    get_video_info,
    run_ffmpeg_command,
    trim_video,
)
from watw.utils.common.validation_utils import (
    ValidationError,
    validate_api_key,
    validate_directory_exists,
    validate_file_exists,
    validate_file_extension,
    validate_required_fields,
)

__all__ = [
    # Validation utilities
    "ValidationError",
    "validate_file_exists",
    "validate_directory_exists",
    "validate_required_fields",
    "validate_file_extension",
    "validate_api_key",
    # Logging utilities
    "setup_logger",
    "log_execution_time",
    "log_function_call",
    # File utilities
    "ensure_directory",
    "get_file_extension",
    "create_temp_file",
    "copy_file",
    # API utilities
    "APIError",
    "APIResponseError",
    "APITimeoutError",
    "TaskStatus",
    "api_config",
    # Media utilities
    "MediaError",
    "FFmpegError",
    "run_ffmpeg_command",
    "trim_video",
    "combine_video_with_audio",
    "add_background_music",
    "concatenate_videos",
    "extract_audio",
    "get_video_info",
]
