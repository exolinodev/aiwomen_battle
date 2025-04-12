"""
Common video operations.

This module provides utility functions for common video operations,
including trimming, concatenation, and audio extraction.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Union, cast, Any

from watw.utils.common.logging_utils import setup_logger
from watw.utils.video.ffmpeg import FFmpegError, run_ffmpeg_command
from watw.utils.video.metadata import VideoValidationError, validate_video_file

logger = setup_logger("watw.video_operations")


def trim_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    duration: float,
    **kwargs: Any,
) -> None:
    """Trim a video file to a specific duration.

    Args:
        input_path: Path to input video file
        output_path: Path to save trimmed video
        start_time: Start time in seconds
        duration: Duration in seconds
        **kwargs: Additional FFmpeg options

    Raises:
        FFmpegError: If trimming fails
        VideoValidationError: If video validation fails
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Validate input video with comprehensive checks
    try:
        video_info = validate_video_file(
            input_path, min_duration=start_time + duration, check_corruption=True
        )
        logger.info(
            f"Validated input video: {video_info['path']} ({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})"
        )
    except VideoValidationError as e:
        logger.error(f"Video validation failed: {str(e)}")
        raise

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-ss",
            str(start_time),
            "-t",
            str(duration),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(output_path),
        ]

        logger.info(f"Trimming video: {input_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = run_ffmpeg_command(cmd)

        if result.returncode != 0:
            raise FFmpegError("Failed to trim video", cmd, result.stderr)

        logger.info("Video trimming completed successfully")

    except Exception as e:
        logger.error(f"Error trimming video: {str(e)}")
        raise FFmpegError("Failed to trim video", cmd, str(e))


def concatenate_videos(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    working_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Concatenate multiple videos into a single video.

    Args:
        video_paths: List of paths to the videos to concatenate
        output_path: Path to save the concatenated video
        working_dir: Working directory for temporary files

    Raises:
        FFmpegError: If the concatenation fails
        VideoValidationError: If any video validation fails
    """
    # Convert paths to Path objects early and ensure it's done
    video_paths_resolved: List[Path] = [Path(p).resolve() for p in video_paths]
    output_path = Path(output_path)

    # Validate all input videos with comprehensive checks
    first_video_info = None
    for video_path in video_paths_resolved:
        try:
            video_info = validate_video_file(video_path, check_corruption=True)

            # Store first video info for dimension checks
            if first_video_info is None:
                first_video_info = video_info
            else:
                # Ensure all videos have the same dimensions
                if (
                    video_info["width"] != first_video_info["width"]
                    or video_info["height"] != first_video_info["height"]
                ):
                    raise VideoValidationError(
                        video_path,
                        f"Video dimensions ({video_info['width']}x{video_info['height']}) "
                        f"do not match first video ({first_video_info['width']}x{first_video_info['height']})",
                    )

            logger.info(
                f"Validated input video: {video_info['path']} "
                f"({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})"
            )

        except VideoValidationError as e:
            logger.error(f"Video validation failed: {str(e)}")
            raise

    # Create working directory if not provided
    if working_dir is None:
        working_dir = Path(tempfile.gettempdir()) / "watw_concat"
    else:
        working_dir = Path(working_dir)

    working_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create a file list for FFmpeg
        file_list_path = working_dir / "file_list.txt"
        with open(file_list_path, "w") as f:
            for video_path in video_paths_resolved:
                f.write(f"file '{str(video_path)}'\n")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(file_list_path),
            "-c",
            "copy",
            str(output_path),
        ]

        logger.info(f"Concatenating {len(video_paths)} videos -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        result = run_ffmpeg_command(cmd)

        if result.returncode != 0:
            raise FFmpegError("Failed to concatenate videos", cmd, result.stderr)

        logger.info("Video concatenation completed successfully")

    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        raise FFmpegError("Failed to concatenate videos", cmd, str(e))
    finally:
        # Clean up temporary file
        if file_list_path.exists():
            file_list_path.unlink()


def extract_audio(
    video_path: Union[str, Path], output_path: Union[str, Path], format: str = "mp3"
) -> Path:
    """Extract audio from a video file.

    Args:
        video_path: Path to the video file
        output_path: Path to save the extracted audio
        format: Audio format (mp3, wav, etc.)

    Returns:
        Path to the extracted audio

    Raises:
        FFmpegError: If the extraction fails
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    # Validate input file
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-acodec",
        "libmp3lame" if format == "mp3" else "pcm_s16le",
        str(output_path),
    ]

    # Run command
    run_ffmpeg_command(cmd)

    return output_path


def combine_video_with_audio(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs: Any,
) -> None:
    """Combine a video file with an audio file.

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

    # Validate input files
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(output_path),
    ]

    # Add volume adjustment if specified
    if "volume" in kwargs:
        cmd.extend(["-filter:a", f"volume={kwargs['volume']}"])

    # Run command
    run_ffmpeg_command(cmd)
