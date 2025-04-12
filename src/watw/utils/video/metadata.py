"""
Video metadata handling utilities.

This module provides utility functions for working with video metadata,
including extraction, validation, and manipulation of video properties.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from watw.utils.common.logging_utils import setup_logger
from watw.utils.video.ffmpeg import (
    FFmpegError,
    build_get_video_info_command,
    run_ffmpeg_command,
)

logger = setup_logger("watw.video_metadata")


@dataclass
class VideoMetadata:
    """Video metadata container."""

    duration: float
    width: int
    height: int
    fps: float
    audio_channels: int
    audio_sample_rate: int
    codec: str = ""
    format: str = ""
    size: int = 0
    bitrate: int = 0


class VideoValidationError(Exception):
    """Exception raised when video validation fails."""

    def __init__(self, video_path: Union[str, Path], message: str):
        self.video_path = video_path
        super().__init__(f"Video validation failed for {video_path}: {message}")


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing video information

    Raises:
        FFmpegError: If the operation fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = build_get_video_info_command(video_path)
    result = run_ffmpeg_command(cmd)

    info = json.loads(result.stdout)

    # Extract video stream information
    video_stream = None
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if not video_stream:
        raise FFmpegError("No video stream found in file")

    # Parse frame rate
    frame_rate_str = video_stream.get("r_frame_rate", "0/1")
    num, den = map(int, frame_rate_str.split("/"))
    frame_rate = num / den

    return {
        "path": str(video_path),
        "duration": float(info.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", ""),
        "format": info.get("format", {}).get("format_name", ""),
        "size": int(info.get("format", {}).get("size", 0)),
        "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
        "fps": frame_rate,
    }


def validate_video_file(
    video_path: Union[str, Path],
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    required_codecs: Optional[List[str]] = None,
    allowed_formats: Optional[List[str]] = None,
    check_corruption: bool = True,
) -> Dict[str, Any]:
    """Perform comprehensive validation of a video file.

    This function checks:
    1. File existence
    2. File format (if allowed_formats is provided)
    3. Video properties (duration, dimensions)
    4. Video codec (if required_codecs is provided)
    5. File corruption (if check_corruption is True)

    Args:
        video_path: Path to the video file
        min_duration: Minimum required duration in seconds
        max_duration: Maximum allowed duration in seconds
        min_width: Minimum required width in pixels
        min_height: Minimum required height in pixels
        required_codecs: List of required video codecs
        allowed_formats: List of allowed file formats (extensions)
        check_corruption: Whether to check for file corruption

    Returns:
        Dict containing video information if validation passes

    Raises:
        VideoValidationError: If any validation check fails
        FFmpegError: If FFmpeg operations fail
    """
    video_path = Path(video_path)

    # Check if file exists
    if not video_path.exists():
        raise VideoValidationError(video_path, "File does not exist")

    # Check file format if specified
    if allowed_formats:
        if video_path.suffix.lower() not in allowed_formats:
            raise VideoValidationError(
                video_path,
                f"File format {video_path.suffix} not in allowed formats: {allowed_formats}",
            )

    # Get video information
    try:
        info = get_video_info(video_path)

        # Check duration
        if min_duration is not None and info["duration"] < min_duration:
            raise VideoValidationError(
                video_path,
                f"Video duration ({info['duration']:.2f}s) is shorter than minimum required ({min_duration}s)",
            )

        if max_duration is not None and info["duration"] > max_duration:
            raise VideoValidationError(
                video_path,
                f"Video duration ({info['duration']:.2f}s) is longer than maximum allowed ({max_duration}s)",
            )

        # Check dimensions
        if min_width is not None and info["width"] < min_width:
            raise VideoValidationError(
                video_path,
                f"Video width ({info['width']}px) is smaller than minimum required ({min_width}px)",
            )

        if min_height is not None and info["height"] < min_height:
            raise VideoValidationError(
                video_path,
                f"Video height ({info['height']}px) is smaller than minimum required ({min_height}px)",
            )

        # Check codec if specified
        if required_codecs and info["codec"] not in required_codecs:
            raise VideoValidationError(
                video_path,
                f"Video codec '{info['codec']}' is not in the list of required codecs: {required_codecs}",
            )

        # Check for corruption by attempting to read a frame
        if check_corruption:
            try:
                # Use ffmpeg to read the first frame without saving it
                corruption_cmd = [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(video_path),
                    "-f",
                    "null",
                    "-frames:v",
                    "1",
                    "-",
                ]
                corruption_result = subprocess.run(
                    corruption_cmd, capture_output=True, text=True
                )

                if corruption_result.returncode != 0:
                    raise VideoValidationError(
                        video_path,
                        f"Video file appears to be corrupted: {corruption_result.stderr}",
                    )
            except Exception as e:
                raise VideoValidationError(
                    video_path, f"Error checking for corruption: {str(e)}"
                )

        return info

    except VideoValidationError:
        raise
    except Exception as e:
        raise VideoValidationError(video_path, f"Error validating video: {str(e)}")


def get_video_duration(video_path: Union[str, Path]) -> float:
    """Get the duration of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        FFmpegError: If the operation fails
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError("Failed to get video duration", cmd, result.stderr)

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise FFmpegError("Invalid duration value", cmd, result.stdout)
