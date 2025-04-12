"""
Video utilities package for Women Around The World.

This package provides a unified set of utilities for video processing,
including FFmpeg operations, metadata handling, and common video operations.
"""

from watw.utils.video.ffmpeg import (
    FFmpegError,
    build_ensure_shorts_resolution_command,
    build_ffmpeg_command,
    build_final_mux_command,
    build_get_video_info_command,
    run_ffmpeg_command,
)
from watw.utils.video.metadata import (
    VideoMetadata,
    VideoValidationError,
    get_video_duration,
    get_video_info,
    validate_video_file,
)
from watw.utils.video.operations import (
    combine_video_with_audio,
    concatenate_videos,
    extract_audio,
    trim_video,
)

__all__ = [
    # FFmpeg utilities
    "FFmpegError",
    "build_ffmpeg_command",
    "build_get_video_info_command",
    "build_ensure_shorts_resolution_command",
    "build_final_mux_command",
    "run_ffmpeg_command",
    # Metadata utilities
    "VideoMetadata",
    "VideoValidationError",
    "get_video_info",
    "validate_video_file",
    "get_video_duration",
    # Operation utilities
    "trim_video",
    "concatenate_videos",
    "extract_audio",
    "combine_video_with_audio",
]
