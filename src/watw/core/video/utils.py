"""
Common video utilities for Women Around The World.

This module provides utility functions for video processing that are used
by the various video editor classes.
"""

import os
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from watw.utils.common.logging_utils import setup_logger
from watw.utils.video.ffmpeg import (
    build_get_video_info_command,
)

logger = setup_logger("watw.video_utils")


class VideoFile:
    """Represents a video file."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.duration: Optional[float] = None
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None

    def get_metadata(self) -> Dict[str, Optional[Union[float, int]]]:
        """Get video metadata."""
        info = get_video_info(self.path)
        if info:
            self.duration = cast(float, info.get("duration"))
            self.width = cast(int, info.get("width"))
            self.height = cast(int, info.get("height"))
            self.fps = cast(float, info.get("fps"))
        return {
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


class AudioFile:
    """Represents an audio file."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        self.duration: Optional[float] = None
        self.sample_rate: Optional[int] = None
        self.channels: Optional[int] = None

    def get_duration(self) -> float:
        """Get audio duration in seconds."""
        if self.duration is None:
            self.duration = get_audio_duration(self.path)
        return self.duration

    def adjust_volume(self, volume: float, output_path: Union[str, Path]) -> None:
        """Adjust audio volume."""
        output_path = Path(output_path)
        cmd = build_ffmpeg_command(
            input_paths=self.path, output_path=output_path, filters=[f"volume={volume}"]
        )
        run_ffmpeg_command(cmd)


class VideoOperationError(Exception):
    """Base exception for video operations."""

    pass


class VideoValidationError(VideoOperationError):
    """Exception raised when video validation fails."""

    pass


class FFmpegError(VideoOperationError):
    """Exception raised when FFmpeg operations fail."""

    pass


class VideoFormat(Enum):
    """Supported video formats."""

    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"


@dataclass
class VideoMetadata:
    """Metadata for a video file."""

    duration: float
    width: int
    height: int
    format: str
    codec: str
    bitrate: int
    fps: float
    size: int


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Get video information using FFmpeg."""
    video_path = Path(video_path)
    cmd = build_get_video_info_command(video_path)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Parse FFmpeg output and return metadata
        # This is a simplified version - you'll need to implement proper parsing
        return {
            "duration": 0.0,
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "format": "",
            "codec": "",
            "bitrate": 0,
            "size": 0,
        }
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to get video info: {e}")


def get_video_duration(path: Union[str, Path]) -> float:
    """Get video duration in seconds."""
    info = get_video_info(path)
    duration = info.get("duration")
    if duration is None:
        raise VideoValidationError(f"Could not determine duration for {path}")
    return float(duration)


def get_audio_duration(path: Union[str, Path]) -> float:
    """Get audio duration in seconds."""
    cmd = build_ffmpeg_command(input_paths=path, output_path=os.devnull, filters=None)
    try:
        result = subprocess.run(
            cmd + ["-f", "null"], capture_output=True, text=True, check=True
        )
        # Parse duration from FFmpeg output
        return 0.0  # Placeholder - implement proper parsing
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Failed to get audio duration: {e}")


def build_ffmpeg_command(
    input_paths: Union[str, Path, Sequence[Union[str, Path]]],
    output_path: Union[str, Path],
    filters: Optional[List[str]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 23,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build FFmpeg command for video processing.

    Args:
        input_paths: Single input path or sequence of input paths
        output_path: Output file path
        filters: List of FFmpeg filters to apply
        codec: Video codec to use
        preset: FFmpeg preset (affects encoding speed vs quality)
        crf: Constant Rate Factor (affects quality)
        pix_fmt: Pixel format

    Returns:
        List of command arguments for FFmpeg
    """
    cmd = ["ffmpeg"]

    # Handle single path or sequence of paths
    if isinstance(input_paths, (str, Path)):
        input_paths = [input_paths]

    # Add input files
    for path in input_paths:
        cmd.extend(["-i", str(path)])

    # Add filters if specified
    if filters:
        cmd.extend(["-filter_complex", ";".join(filters)])

    # Add encoding options
    cmd.extend(
        [
            "-c:v",
            codec,
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pix_fmt,
            "-y",  # Overwrite output file
            str(output_path),
        ]
    )

    return cmd


def run_ffmpeg_command(cmd: List[str], timeout: int = 300) -> None:
    """Run FFmpeg command with timeout."""
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FFmpegError("FFmpeg command timed out")
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"FFmpeg command failed: {e}")


def create_temp_file(prefix: str = "watw_", suffix: str = ".mp4") -> Path:
    """Create a temporary file."""
    import tempfile

    return Path(tempfile.mktemp(prefix=prefix, suffix=suffix))


def trim_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    duration: float,
) -> None:
    """Trim video to specified duration."""
    input_path = str(input_path)
    output_path = str(output_path)

    cmd = [
        "ffmpeg",
        "-i",
        input_path,
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-c",
        "copy",
        "-y",
        output_path,
    ]

    run_ffmpeg_command(cmd)


def validate_video_file(path: Union[str, Path]) -> bool:
    """Validate video file."""
    try:
        get_video_info(path)
        return True
    except FFmpegError:
        return False
