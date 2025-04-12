"""Core video utility functions for the Women Around The World project."""

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence

from watw.utils.common.media_utils import (
    FFmpegError,
    VideoValidationError,
    build_combine_video_audio_command,
    build_get_video_info_command,
    build_trim_command,
    run_ffmpeg_command,
    validate_directory_exists,
    validate_file_exists,
    validate_file_extension,
)

logger = logging.getLogger(__name__)


def get_video_duration(video_path: Union[str, Path]) -> float:
    """
    Get the duration of a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Duration in seconds

    Raises:
        FFmpegError: If the operation fails
    """
    video_path = Path(video_path)
    validate_file_exists(video_path)

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
        raise FFmpegError(f"Failed to get video duration: {result.stderr}")

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise FFmpegError(f"Invalid duration value: {result.stdout}")


def get_audio_duration(audio_path: Union[str, Path]) -> float:
    """
    Get the duration of an audio file.

    Args:
        audio_path: Path to the audio file

    Returns:
        Duration in seconds

    Raises:
        FFmpegError: If the operation fails
    """
    return get_video_duration(audio_path)  # FFprobe works the same way for audio files


def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a video file.

    Args:
        video_path: Path to the video file

    Returns:
        Dictionary containing video information

    Raises:
        FFmpegError: If the operation fails
    """
    video_path = Path(video_path)
    validate_file_exists(video_path)

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

    return {
        "path": str(video_path),
        "duration": float(info.get("format", {}).get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", ""),
        "format": info.get("format", {}).get("format_name", ""),
        "size": int(info.get("format", {}).get("size", 0)),
        "bitrate": int(info.get("format", {}).get("bit_rate", 0)),
        "fps": eval(video_stream.get("r_frame_rate", "0/1")),
    }


def validate_video_corruption(video_path: Union[str, Path]) -> None:
    """
    Check if a video file is corrupted by attempting to read its metadata.

    Args:
        video_path: Path to the video file

    Raises:
        FFmpegError: If the video appears to be corrupted
    """
    video_path = Path(video_path)
    validate_file_exists(video_path)

    # Use ffprobe to check for corruption
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "json",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"Video appears to be corrupted: {result.stderr}")

    try:
        info = json.loads(result.stdout)
        if not info.get("streams"):
            raise FFmpegError("No video streams found in file")
    except json.JSONDecodeError:
        raise FFmpegError("Failed to parse video metadata")


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
    """
    Perform comprehensive validation of a video file.

    Args:
        video_path: Path to the video file
        min_duration: Minimum required duration in seconds
        max_duration: Maximum allowed duration in seconds
        min_width: Minimum required width in pixels
        min_height: Minimum required height in pixels
        required_codecs: List of required video codecs
        allowed_formats: List of allowed file formats
        check_corruption: Whether to check for file corruption

    Returns:
        Dict containing video information if validation passes

    Raises:
        VideoValidationError: If any validation check fails
        FFmpegError: If FFmpeg operations fail
    """
    video_path = Path(video_path)

    # Basic file validation
    validate_file_exists(video_path)
    if allowed_formats:
        validate_file_extension(video_path, allowed_formats)

    # Get video information
    info = get_video_info(video_path)

    # Validate duration
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

    # Validate dimensions
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

    # Validate codecs
    if required_codecs:
        missing_codecs = [
            codec for codec in required_codecs if codec not in info["codecs"]
        ]
        if missing_codecs:
            raise VideoValidationError(
                video_path,
                f"Video is missing required codecs: {', '.join(missing_codecs)}",
            )

    # Check for corruption
    if check_corruption:
        try:
            validate_video_corruption(video_path)
        except FFmpegError as e:
            raise VideoValidationError(
                video_path, f"Video appears to be corrupted: {str(e)}"
            )

    return info


def combine_video_with_audio(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs: Any,
) -> None:
    """
    Combine a video file with an audio file.

    Args:
        video_path: Path to input video file
        audio_path: Path to input audio file
        output_path: Path to save combined video
        **kwargs: Additional FFmpeg options

    Raises:
        FFmpegError: If combining fails
        VideoValidationError: If video validation fails
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    output_path = Path(output_path)

    # Validate inputs
    validate_video_file(video_path, check_corruption=True)
    validate_file_exists(audio_path)
    validate_directory_exists(output_path.parent)

    try:
        cmd = build_combine_video_audio_command(
            video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            **kwargs,
        )

        logger.info(
            f"Combining video and audio: {video_path} + {audio_path} -> {output_path}"
        )
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise FFmpegError(f"Failed to combine video and audio: {result.stderr}")

        logger.info("Video and audio combination completed successfully")

    except Exception as e:
        logger.error(f"Error combining video and audio: {str(e)}")
        raise FFmpegError(f"Failed to combine video and audio: {str(e)}")


def concatenate_videos(
    video_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    working_dir: Optional[Union[str, Path]] = None,
    validate_dimensions: bool = True,
) -> None:
    """
    Concatenate multiple video files into a single video.

    Args:
        video_paths: List of paths to video files to concatenate
        output_path: Path where the concatenated video will be saved
        working_dir: Optional directory for temporary files
        validate_dimensions: Whether to validate video dimensions match

    Raises:
        VideoValidationError: If video validation fails
        FFmpegError: If FFmpeg operations fail
    """
    if not video_paths:
        raise ValueError("No video paths provided")

    # Convert all paths to Path objects
    video_paths = [Path(p) for p in video_paths]
    output_path = Path(output_path)
    working_dir = Path(working_dir) if working_dir else Path(tempfile.mkdtemp())

    # Validate input files
    for video_path in video_paths:
        validate_file_exists(video_path)
        validate_file_extension(video_path, [".mp4", ".mov", ".mkv"])

    # Create working directory if it doesn't exist
    working_dir.mkdir(parents=True, exist_ok=True)

    # Create a file list for FFmpeg
    file_list_path = working_dir / "file_list.txt"
    with open(file_list_path, "w") as f:
        for video_path in video_paths:
            # Convert to Path and resolve to get absolute path
            resolved_path = Path(video_path).resolve()
            f.write(f"file '{resolved_path}'\n")

    # Use FFmpeg to concatenate videos
    cmd = [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c",
        "copy",
        "-y",
        str(output_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise FFmpegError(f"Failed to concatenate videos: {result.stderr}")

        logger.info("Video concatenation completed successfully")

    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        raise FFmpegError(f"Failed to concatenate videos: {str(e)}")


def trim_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    duration: float,
    **kwargs: Any,
) -> None:
    """
    Trim a video file to a specific duration starting from a given time.

    Args:
        input_path: Path to the input video file
        output_path: Path where the trimmed video will be saved
        start_time: Start time in seconds
        duration: Duration to trim in seconds
        **kwargs: Additional arguments to pass to FFmpeg

    Raises:
        VideoValidationError: If video validation fails
        FFmpegError: If FFmpeg operations fail
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input file
    validate_file_exists(input_path)
    validate_file_extension(input_path, [".mp4", ".mov", ".mkv"])

    # Calculate end time
    end_time = start_time + duration

    # Build and run FFmpeg command
    cmd = build_trim_command(
        input_path=str(input_path),
        output_path=str(output_path),
        start_time=start_time,
        end_time=end_time,
        **kwargs,
    )

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise FFmpegError(f"Failed to trim video: {result.stderr}")

        logger.info("Video trimming completed successfully")

    except Exception as e:
        logger.error(f"Error trimming video: {str(e)}")
        raise FFmpegError(f"Failed to trim video: {str(e)}")
