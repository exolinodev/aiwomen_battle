"""
Common utility modules for Women Around The World.

This package provides common utility functions and classes used throughout
the Women Around The World application.
"""

from watw.utils.common.validation_utils import (
    ValidationError,
    validate_file_exists,
    validate_directory_exists,
    validate_required_fields,
    validate_file_extension,
    validate_api_key
)

from watw.utils.common.logging_utils import (
    setup_logger,
    log_execution_time,
    log_function_call
)

from watw.utils.common.file_utils import (
    ensure_directory,
    get_file_extension,
    create_temp_file,
    copy_file
)

from watw.utils.common.api_utils import (
    APIError,
    APIResponseError,
    APITimeoutError,
    TaskStatus,
    APIConfiguration,
    api_config,
    check_api_credentials,
    download_file as api_download_file,
    encode_image_base64,
    poll_api_task,
    get_runway_client
)

from watw.utils.common.media_utils import (
    MediaError,
    FFmpegError,
    run_ffmpeg_command,
    trim_video,
    combine_video_with_audio,
    add_background_music,
    concatenate_videos,
    extract_audio,
    get_video_info
)

from .mock_api_client import MockTensorArtClient, MockRunwayMLClient
from .mock_api_responses import mock_responses
from .test_utils import MockAPITestCase

__all__ = [
    # Validation utilities
    'ValidationError',
    'validate_file_exists',
    'validate_directory_exists',
    'validate_required_fields',
    'validate_file_extension',
    'validate_api_key',
    
    # Logging utilities
    'setup_logger',
    'log_execution_time',
    'log_function_call',
    
    # File utilities
    'ensure_directory',
    'get_file_extension',
    'create_temp_file',
    'copy_file',
    
    # API utilities
    'APIError',
    'APIResponseError',
    'APITimeoutError',
    'TaskStatus',
    'APIConfiguration',
    'api_config',
    'check_api_credentials',
    'api_download_file',
    'encode_image_base64',
    'poll_api_task',
    'get_runway_client',
    
    # Media utilities
    'MediaError',
    'FFmpegError',
    'run_ffmpeg_command',
    'trim_video',
    'combine_video_with_audio',
    'add_background_music',
    'concatenate_videos',
    'extract_audio',
    'get_video_info',

    'MockTensorArtClient',
    'MockRunwayMLClient',
    'mock_responses',
    'MockAPITestCase'
] 