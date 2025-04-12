"""
FFmpeg-specific utilities for video processing.

This module provides utility functions for working with FFmpeg commands
and operations, including command building and execution.
"""

import subprocess
from pathlib import Path
from typing import List, Optional, Union, Sequence

from watw.utils.common.logging_utils import setup_logger

logger = setup_logger("watw.ffmpeg")


class FFmpegError(Exception):
    """Exception raised when an FFmpeg operation fails."""

    def __init__(
        self,
        message: str,
        command: Optional[List[str]] = None,
        error: Optional[str] = None,
    ):
        self.command = command
        self.error = error
        super().__init__(
            f"{message}\nCommand: {' '.join(command) if command else 'N/A'}\nError: {error if error else 'N/A'}"
        )


def build_ffmpeg_command(
    input_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    filters: Optional[List[str]] = None,
    codec: str = "libx264",
    preset: str = "medium",
    crf: int = 23,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build an FFmpeg command for video processing.

    Args:
        input_paths: List of input file paths
        output_path: Output file path
        filters: Optional list of FFmpeg filters
        codec: Video codec to use
        preset: Encoding preset
        crf: Constant Rate Factor (quality)
        pix_fmt: Pixel format

    Returns:
        List of command arguments
    """
    cmd = ["ffmpeg", "-y"]

    # Add input files
    for path in input_paths:
        cmd.extend(["-i", str(path)])

    # Add filters if provided
    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    # Add encoding parameters
    cmd.extend(
        ["-c:v", codec, "-preset", preset, "-crf", str(crf), "-pix_fmt", pix_fmt]
    )

    # Add output path
    cmd.append(str(output_path))

    return cmd


def build_get_video_info_command(video_path: Union[str, Path]) -> List[str]:
    """Build FFprobe command for getting video information.

    Args:
        video_path: Path to video file

    Returns:
        List of command arguments
    """
    return [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,bit_rate",
        "-show_entries",
        "stream=width,height,codec_name,codec_type",
        "-of",
        "json",
        str(video_path),
    ]


def build_ensure_shorts_resolution_command(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_width: int = 1080,
    target_height: int = 1920,
    video_codec: str = "libx264",
    preset: str = "medium",
    bitrate: str = "4M",
    fps: int = 30,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build FFmpeg command for ensuring video is in Shorts resolution (9:16).

    Args:
        input_path: Path to input video
        output_path: Path to save resized video
        target_width: Target width in pixels
        target_height: Target height in pixels
        video_codec: Video codec to use
        preset: Encoding preset
        bitrate: Video bitrate
        fps: Target frame rate
        pix_fmt: Pixel format

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-i",
        str(input_path),
        "-vf",
        f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        "2M",
        "-pix_fmt",
        pix_fmt,
        "-r",
        str(fps),
        "-c:a",
        "copy",
        str(output_path),
        "-y",
    ]


def build_final_mux_command(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    duration: Optional[float] = None,
    video_fade: bool = False,
    video_filter: Optional[str] = None,
    audio_fade: bool = False,
    audio_filter: Optional[str] = None,
    video_codec: str = "libx264",
    preset: str = "medium",
    bitrate: str = "4M",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    fps: int = 30,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build FFmpeg command for final video muxing with audio.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path to save final video
        duration: Target duration in seconds
        video_fade: Whether to apply video fade
        video_filter: Video filter string
        audio_fade: Whether to apply audio fade
        audio_filter: Audio filter string
        video_codec: Video codec to use
        preset: Encoding preset
        bitrate: Video bitrate
        audio_codec: Audio codec to use
        audio_bitrate: Audio bitrate
        fps: Target frame rate
        pix_fmt: Pixel format

    Returns:
        List of command arguments
    """
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
    ]

    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])

    if video_fade and video_filter:
        cmd.extend(["-vf", video_filter])
        cmd.extend(
            [
                "-c:v",
                video_codec,
                "-preset",
                preset,
                "-b:v",
                bitrate,
                "-maxrate",
                bitrate,
                "-bufsize",
                "2M",
                "-pix_fmt",
                pix_fmt,
                "-r",
                str(fps),
            ]
        )
    else:
        cmd.extend(["-c:v", "copy"])

    cmd.extend(["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", "2", "-ar", "48000"])

    if audio_fade and audio_filter:
        cmd.extend(["-af", audio_filter])

    cmd.extend([str(output_path), "-y"])
    return cmd


def build_trim_command(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    end_time: float,
    video_codec: str = "libx264",
    preset: str = "medium",
    bitrate: str = "4M",
    fps: int = 30,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build FFmpeg command for trimming a video.

    Args:
        input_path: Path to input video
        output_path: Path to save trimmed video
        start_time: Start time in seconds
        end_time: End time in seconds
        video_codec: Video codec to use
        preset: Encoding preset
        bitrate: Video bitrate
        fps: Target frame rate
        pix_fmt: Pixel format

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-i",
        str(input_path),
        "-ss",
        f"{start_time:.3f}",
        "-to",
        f"{end_time:.3f}",
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        "2M",
        "-pix_fmt",
        pix_fmt,
        "-r",
        str(fps),
        "-c:a",
        "copy",
        str(output_path),
        "-y",
    ]


def build_combine_video_audio_command(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> List[str]:
    """Build FFmpeg command for combining video and audio.

    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path to save combined video
        video_codec: Video codec to use
        audio_codec: Audio codec to use
        audio_bitrate: Audio bitrate

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        str(output_path),
        "-y",
    ]


def build_add_background_music_command(
    video_path: Union[str, Path],
    music_path: Union[str, Path],
    output_path: Union[str, Path],
    music_volume: float = 0.1,
    video_volume: float = 1.0,
    video_codec: str = "copy",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> List[str]:
    """Build FFmpeg command for adding background music to a video.

    Args:
        video_path: Path to video file
        music_path: Path to music file
        output_path: Path to save output video
        music_volume: Volume level for background music (0.0 to 1.0)
        video_volume: Volume level for video audio (0.0 to 1.0)
        video_codec: Video codec to use
        audio_codec: Audio codec to use
        audio_bitrate: Audio bitrate

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-i",
        str(video_path),
        "-i",
        str(music_path),
        "-filter_complex",
        f"[0:a]volume={video_volume}[a1];[1:a]volume={music_volume}[a2];[a1][a2]amix=inputs=2:duration=first[aout]",
        "-map",
        "0:v",
        "-map",
        "[aout]",
        "-c:v",
        video_codec,
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        str(output_path),
        "-y",
    ]


def build_concatenate_videos_command(
    file_list_path: Union[str, Path],
    output_path: Union[str, Path],
    video_codec: str = "libx264",
    preset: str = "medium",
    bitrate: str = "4M",
    fps: int = 30,
    pix_fmt: str = "yuv420p",
) -> List[str]:
    """Build FFmpeg command for concatenating videos using a file list.

    Args:
        file_list_path: Path to file containing list of input videos
        output_path: Path to save concatenated video
        video_codec: Video codec to use
        preset: Encoding preset
        bitrate: Video bitrate
        fps: Target frame rate
        pix_fmt: Pixel format

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(file_list_path),
        "-c:v",
        video_codec,
        "-preset",
        preset,
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        "2M",
        "-pix_fmt",
        pix_fmt,
        "-r",
        str(fps),
        "-c:a",
        "copy",
        str(output_path),
        "-y",
    ]


def build_extract_audio_command(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> List[str]:
    """Build FFmpeg command for extracting audio from a video.

    Args:
        video_path: Path to input video
        output_path: Path to save extracted audio
        audio_codec: Audio codec to use
        audio_bitrate: Audio bitrate

    Returns:
        List of command arguments
    """
    return [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",  # No video
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
        str(output_path),
        "-y",
    ]


def run_ffmpeg_command(
    cmd: List[str], timeout: int = 300
) -> subprocess.CompletedProcess:
    """Run an FFmpeg command.

    Args:
        cmd: List of command arguments
        timeout: Command timeout in seconds

    Returns:
        CompletedProcess object

    Raises:
        FFmpegError: If the command fails
        subprocess.TimeoutExpired: If the command times out
    """
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg command failed: {e}")
        raise FFmpegError("FFmpeg command failed", cmd, e.stderr)
    except subprocess.TimeoutExpired:
        logger.error(f"FFmpeg command timed out after {timeout} seconds")
        raise FFmpegError(f"FFmpeg command timed out after {timeout} seconds", cmd)
