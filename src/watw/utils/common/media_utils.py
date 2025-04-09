"""
Media utility functions for Women Around The World.

This module provides utility functions for working with media files,
including video, audio, and image processing. It centralizes media-related
functionality to reduce code duplication and improve maintainability.
"""

# Standard library imports
import os
import subprocess
import tempfile
import json
import enum
import numpy as np
from pathlib import Path
from typing import Union, Optional, List, Dict, Any, Tuple, Sequence

# Third-party imports
import librosa
import soundfile as sf

# Local imports
from watw.utils.common.validation_utils import validate_file_exists, validate_directory_exists, validate_file_extension
from watw.utils.common.logging_utils import setup_logger
from watw.utils.common.ffmpeg_utils import (
    build_trim_command,
    build_combine_video_audio_command,
    build_add_background_music_command,
    build_concatenate_videos_command,
    build_extract_audio_command,
    build_get_video_info_command,
    build_ensure_shorts_resolution_command,
    build_final_mux_command
)

# Set up logger
logger = setup_logger("watw.media")

class TransitionType(enum.Enum):
    """
    Supported video transition types.
    """
    CUT = "cut"  # Direct cut (no transition)
    CROSSFADE = "crossfade"  # Fade between clips
    FADE_TO_BLACK = "fade_to_black"  # Fade through black
    FADE_TO_WHITE = "fade_to_white"  # Fade through white
    WIPE_LEFT = "wipe_left"  # Wipe from right to left
    WIPE_RIGHT = "wipe_right"  # Wipe from left to right
    WIPE_UP = "wipe_up"  # Wipe from bottom to top
    WIPE_DOWN = "wipe_down"  # Wipe from top to bottom
    ZOOM_IN = "zoom_in"  # Zoom into first clip
    ZOOM_OUT = "zoom_out"  # Zoom out from second clip
    BLUR = "blur"  # Blur transition
    DISSOLVE = "dissolve"  # Dissolve between clips

class TransitionConfig:
    """
    Configuration for video transitions.
    
    Attributes:
        type: Type of transition
        duration: Duration of transition in seconds
        params: Additional parameters for the transition
    """
    def __init__(
        self,
        type: Union[TransitionType, str],
        duration: float = 1.0,
        **params
    ):
        self.type = TransitionType(type) if isinstance(type, str) else type
        self.duration = duration
        self.params = params

def build_transition_filter(
    transition: TransitionConfig,
    video1_stream: str = "[0:v]",
    video2_stream: str = "[1:v]",
    output_stream: str = "[v]"
) -> str:
    """
    Build FFmpeg filter complex string for a transition effect.
    
    Args:
        transition: Transition configuration
        video1_stream: First video stream specifier
        video2_stream: Second video stream specifier
        output_stream: Output stream specifier
        
    Returns:
        FFmpeg filter complex string
    """
    duration = transition.duration
    
    if transition.type == TransitionType.CUT:
        return f"{video1_stream}{video2_stream}concat=n=2:v=1:a=0{output_stream}"
        
    elif transition.type == TransitionType.CROSSFADE:
        return (
            f"{video1_stream}format=pix_fmts=yuva420p,fade=t=out:st=0:d={duration}:alpha=1[fade_out];"
            f"{video2_stream}format=pix_fmts=yuva420p,fade=t=in:st=0:d={duration}:alpha=1[fade_in];"
            f"[fade_out][fade_in]overlay=shortest=1{output_stream}"
        )
        
    elif transition.type == TransitionType.FADE_TO_BLACK:
        return (
            f"{video1_stream}fade=t=out:st=0:d={duration}[fade_out];"
            f"{video2_stream}fade=t=in:st=0:d={duration}[fade_in];"
            f"[fade_out][fade_in]concat=n=2:v=1:a=0{output_stream}"
        )
        
    elif transition.type == TransitionType.FADE_TO_WHITE:
        return (
            f"{video1_stream}fade=t=out:st=0:d={duration}:color=white[fade_out];"
            f"{video2_stream}fade=t=in:st=0:d={duration}:color=white[fade_in];"
            f"[fade_out][fade_in]concat=n=2:v=1:a=0{output_stream}"
        )
        
    elif transition.type in (TransitionType.WIPE_LEFT, TransitionType.WIPE_RIGHT):
        direction = "1" if transition.type == TransitionType.WIPE_LEFT else "-1"
        return (
            f"{video1_stream}{video2_stream}xfade=transition=slide:"
            f"duration={duration}:offset=0:slide_x={direction}{output_stream}"
        )
        
    elif transition.type in (TransitionType.WIPE_UP, TransitionType.WIPE_DOWN):
        direction = "1" if transition.type == TransitionType.WIPE_UP else "-1"
        return (
            f"{video1_stream}{video2_stream}xfade=transition=slide:"
            f"duration={duration}:offset=0:slide_y={direction}{output_stream}"
        )
        
    elif transition.type == TransitionType.ZOOM_IN:
        return (
            f"{video1_stream}scale=2*iw:-1,crop=iw/2:ih/2:(iw-iw/2)/2:(ih-ih/2)/2[v1];"
            f"{video2_stream}[v1]blend=all_expr='A*T+B*(1-T)'{output_stream}"
        )
        
    elif transition.type == TransitionType.ZOOM_OUT:
        return (
            f"{video2_stream}scale=2*iw:-1,crop=iw/2:ih/2:(iw-iw/2)/2:(ih-ih/2)/2[v2];"
            f"{video1_stream}[v2]blend=all_expr='A*(1-T)+B*T'{output_stream}"
        )
        
    elif transition.type == TransitionType.BLUR:
        return (
            f"{video1_stream}split[v1][v1blur];"
            f"[v1blur]boxblur=10[v1blurred];"
            f"[v1][v1blurred]blend=all_expr='A*(1-T)+B*T'[v1fade];"
            f"{video2_stream}split[v2][v2blur];"
            f"[v2blur]boxblur=10[v2blurred];"
            f"[v2][v2blurred]blend=all_expr='A*T+B*(1-T)'[v2fade];"
            f"[v1fade][v2fade]blend=all_expr='A*(1-T)+B*T'{output_stream}"
        )
        
    elif transition.type == TransitionType.DISSOLVE:
        return (
            f"{video1_stream}{video2_stream}xfade=transition=dissolve:"
            f"duration={duration}:offset=0{output_stream}"
        )
    
    else:
        raise ValueError(f"Unsupported transition type: {transition.type}")

def concatenate_videos_with_transitions(
    video_paths: Sequence[Union[str, Path]],
    output_path: Union[str, Path],
    transitions: Optional[Sequence[TransitionConfig]] = None,
    working_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Concatenate multiple videos with transitions between them.
    
    Args:
        video_paths: List of paths to the videos to concatenate
        output_path: Path to save the concatenated video
        transitions: List of transitions to apply between videos.
                    If None, uses direct cuts.
                    If provided, must be len(video_paths) - 1.
        working_dir: Working directory for temporary files
        
    Raises:
        FFmpegError: If the concatenation fails
        VideoValidationError: If any video validation fails
        ValueError: If transitions list length is incorrect
    """
    if len(video_paths) < 2:
        raise ValueError("At least two videos are required for concatenation")
    
    if transitions is None:
        transitions = [TransitionConfig(TransitionType.CUT)] * (len(video_paths) - 1)
    elif len(transitions) != len(video_paths) - 1:
        raise ValueError(
            f"Number of transitions ({len(transitions)}) must be one less than "
            f"number of videos ({len(video_paths)})"
        )
    
    # Convert paths to Path objects
    video_paths = [Path(p) for p in video_paths]
    output_path = Path(output_path)
    
    # Validate all input videos
    first_video_info = None
    for video_path in video_paths:
        try:
            video_info = validate_video_file(
                video_path,
                check_corruption=True
            )
            
            if first_video_info is None:
                first_video_info = video_info
            else:
                if (video_info['width'] != first_video_info['width'] or 
                    video_info['height'] != first_video_info['height']):
                    raise VideoValidationError(
                        video_path,
                        f"Video dimensions ({video_info['width']}x{video_info['height']}) "
                        f"do not match first video ({first_video_info['width']}x{first_video_info['height']})"
                    )
            
            logger.info(f"Validated input video: {video_info['path']} "
                       f"({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})")
                       
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
        # Build complex filter for transitions
        filter_complex = ""
        for i, (transition, video_path) in enumerate(zip(transitions, video_paths[1:])):
            v1 = f"[v{i}]" if i > 0 else "[0:v]"
            v2 = f"[{i+1}:v]"
            v_out = f"[v{i+1}]" if i < len(transitions) - 1 else "[v]"
            
            filter_complex += build_transition_filter(
                transition,
                video1_stream=v1,
                video2_stream=v2,
                output_stream=v_out
            )
            if i < len(transitions) - 1:
                filter_complex += ";"
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y']
        
        # Add input files
        for video_path in video_paths:
            cmd.extend(['-i', str(video_path)])
        
        # Add filter complex
        cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[v]',  # Map video output
            str(output_path)
        ])
        
        logger.info(f"Concatenating {len(video_paths)} videos with transitions -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to concatenate videos: {result.stderr}")
            
        logger.info("Video concatenation with transitions completed successfully")
        
    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        raise FFmpegError(f"Failed to concatenate videos: {str(e)}")

class MediaError(Exception):
    """
    Base exception class for media-related errors.
    
    This is the parent class for all media-specific exceptions.
    """
    pass

class FFmpegError(MediaError):
    """
    Exception raised when an FFmpeg operation fails.
    
    Attributes:
        command: The FFmpeg command that failed
        error: The error message
    """
    def __init__(self, command: List[str], error: str):
        self.command = command
        self.error = error
        super().__init__(f"FFmpeg command failed: {' '.join(command)}\nError: {error}")

class VideoValidationError(MediaError):
    """
    Exception raised when video validation fails.
    
    Attributes:
        file_path: Path to the video file that failed validation
        reason: Reason for validation failure
    """
    def __init__(self, file_path: Union[str, Path], reason: str):
        self.file_path = str(file_path)
        self.reason = reason
        super().__init__(f"Video validation failed for {self.file_path}: {reason}")

def run_ffmpeg_command(command: List[str], check: bool = True) -> subprocess.CompletedProcess:
    """
    Run an FFmpeg command and handle errors.
    
    Args:
        command: The FFmpeg command to run
        check: Whether to raise an exception if the command fails
        
    Returns:
        CompletedProcess object
        
    Raises:
        FFmpegError: If the command fails and check is True
    """
    try:
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True
        )
        
        if check and result.returncode != 0:
            raise FFmpegError(command, result.stderr)
        
        return result
    except subprocess.CalledProcessError as e:
        raise FFmpegError(command, e.stderr)

def trim_video(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    start_time: float,
    duration: float,
    **kwargs
) -> None:
    """
    Trim a video file to a specific duration.

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
            input_path,
            min_duration=start_time + duration,
            check_corruption=True
        )
        logger.info(f"Validated input video: {video_info['path']} ({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})")
    except VideoValidationError as e:
        logger.error(f"Video validation failed: {str(e)}")
        raise
    
    # Ensure output directory exists
    validate_directory_exists(output_path.parent)
    
    try:
        cmd = build_trim_command(
            input_path=str(input_path),
            output_path=str(output_path),
            start_time=start_time,
            duration=duration,
            **kwargs
        )
        
        logger.info(f"Trimming video: {input_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to trim video: {result.stderr}")
            
        logger.info("Video trimming completed successfully")
        
    except Exception as e:
        logger.error(f"Error trimming video: {str(e)}")
        raise FFmpegError(f"Failed to trim video: {str(e)}")

def combine_video_with_audio(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
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
    
    # Validate input video with comprehensive checks
    try:
        video_info = validate_video_file(
            video_path,
            check_corruption=True
        )
        logger.info(f"Validated input video: {video_info['path']} ({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})")
    except VideoValidationError as e:
        logger.error(f"Video validation failed: {str(e)}")
        raise
    
    # Validate audio file exists
    validate_file_exists(audio_path)
    validate_directory_exists(output_path.parent)
    
    try:
        cmd = build_combine_video_audio_command(
            video_path=str(video_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            **kwargs
        )
        
        logger.info(f"Combining video and audio: {video_path} + {audio_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to combine video and audio: {result.stderr}")
            
        logger.info("Video and audio combination completed successfully")
        
    except Exception as e:
        logger.error(f"Error combining video and audio: {str(e)}")
        raise FFmpegError(f"Failed to combine video and audio: {str(e)}")

def add_background_music(
    video_path: Union[str, Path],
    music_path: Union[str, Path],
    output_path: Union[str, Path],
    volume: float = 0.1,
    **kwargs
) -> None:
    """
    Add background music to a video with specified volume.

    Args:
        video_path: Path to input video file
        music_path: Path to background music file
        output_path: Path to save output video
        volume: Volume level for background music (0.0 to 1.0)
        **kwargs: Additional FFmpeg options

    Raises:
        FFmpegError: If adding background music fails
    """
    video_path = Path(video_path)
    music_path = Path(music_path)
    output_path = Path(output_path)
    
    validate_file_exists(video_path)
    validate_file_exists(music_path)
    validate_directory_exists(output_path.parent)
    
    if not 0.0 <= volume <= 1.0:
        raise ValueError("Volume must be between 0.0 and 1.0")
    
    try:
        cmd = build_add_background_music_command(
            video_path=str(video_path),
            music_path=str(music_path),
            output_path=str(output_path),
            volume=volume,
            **kwargs
        )
        
        logger.info(f"Adding background music to video: {video_path} + {music_path} -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to add background music: {result.stderr}")
            
        logger.info("Background music added successfully")
        
    except Exception as e:
        logger.error(f"Error adding background music: {str(e)}")
        raise FFmpegError(f"Failed to add background music: {str(e)}")

def concatenate_videos(
    video_paths: List[Union[str, Path]],
    output_path: Union[str, Path],
    working_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Concatenate multiple videos into a single video.
    
    Args:
        video_paths: List of paths to the videos to concatenate
        output_path: Path to save the concatenated video
        working_dir: Working directory for temporary files
        
    Raises:
        FFmpegError: If the concatenation fails
        VideoValidationError: If any video validation fails
    """
    # Convert paths to Path objects
    video_paths = [Path(p) for p in video_paths]
    output_path = Path(output_path)
    
    # Validate all input videos with comprehensive checks
    first_video_info = None
    for video_path in video_paths:
        try:
            video_info = validate_video_file(
                video_path,
                check_corruption=True
            )
            
            # Store first video info for dimension checks
            if first_video_info is None:
                first_video_info = video_info
            else:
                # Ensure all videos have the same dimensions
                if (video_info['width'] != first_video_info['width'] or 
                    video_info['height'] != first_video_info['height']):
                    raise VideoValidationError(
                        video_path,
                        f"Video dimensions ({video_info['width']}x{video_info['height']}) "
                        f"do not match first video ({first_video_info['width']}x{first_video_info['height']})"
                    )
            
            logger.info(f"Validated input video: {video_info['path']} "
                       f"({video_info['duration']:.2f}s, {video_info['width']}x{video_info['height']})")
                       
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
        with open(file_list_path, 'w') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.absolute()}'\n")
        
        cmd = build_concatenate_videos_command(
            file_list_path=str(file_list_path),
            output_path=str(output_path)
        )
        
        logger.info(f"Concatenating {len(video_paths)} videos -> {output_path}")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise FFmpegError(f"Failed to concatenate videos: {result.stderr}")
            
        logger.info("Video concatenation completed successfully")
        
    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        raise FFmpegError(f"Failed to concatenate videos: {str(e)}")
    finally:
        # Clean up temporary file
        if file_list_path.exists():
            file_list_path.unlink()

def extract_audio(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    format: str = "mp3"
) -> Path:
    """
    Extract audio from a video file.
    
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
    validate_file_exists(video_path)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build FFmpeg command
    command = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'libmp3lame' if format == "mp3" else 'pcm_s16le',
        str(output_path)
    ]
    
    # Run command
    run_ffmpeg_command(command)
    
    return output_path

def get_video_info(video_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a video file.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Dictionary containing video information
        
    Raises:
        FFmpegError: If the operation fails
    """
    video_path = Path(video_path)
    
    # Validate input file
    validate_file_exists(video_path)
    
    # Build FFprobe command
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration,size,bit_rate',
        '-show_entries', 'stream=width,height,codec_name,codec_type',
        '-of', 'json',
        str(video_path)
    ]
    
    # Run command
    result = run_ffmpeg_command(command)
    
    # Parse JSON output
    info = json.loads(result.stdout)
    
    return info 

def validate_video_file(
    video_path: Union[str, Path],
    min_duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    required_codecs: Optional[List[str]] = None,
    allowed_formats: Optional[List[str]] = None,
    check_corruption: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive validation of a video file.
    
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
    try:
        validate_file_exists(video_path)
    except Exception as e:
        raise VideoValidationError(video_path, f"File does not exist: {str(e)}")
    
    # Check file format if specified
    if allowed_formats:
        try:
            validate_file_extension(video_path, allowed_formats)
        except Exception as e:
            raise VideoValidationError(video_path, f"Invalid file format: {str(e)}")
    
    # Get video information
    try:
        cmd = build_get_video_info_command(video_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise VideoValidationError(video_path, f"Failed to get video info: {result.stderr}")
        
        # Parse JSON output
        info = json.loads(result.stdout)
        
        # Extract video stream information
        video_stream = None
        for stream in info.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
                break
        
        if not video_stream:
            raise VideoValidationError(video_path, "No video stream found in file")
        
        # Check duration
        duration = float(info.get('format', {}).get('duration', 0))
        if min_duration is not None and duration < min_duration:
            raise VideoValidationError(
                video_path, 
                f"Video duration ({duration:.2f}s) is shorter than minimum required ({min_duration}s)"
            )
        
        if max_duration is not None and duration > max_duration:
            raise VideoValidationError(
                video_path, 
                f"Video duration ({duration:.2f}s) is longer than maximum allowed ({max_duration}s)"
            )
        
        # Check dimensions
        width = int(video_stream.get('width', 0))
        height = int(video_stream.get('height', 0))
        
        if min_width is not None and width < min_width:
            raise VideoValidationError(
                video_path, 
                f"Video width ({width}px) is smaller than minimum required ({min_width}px)"
            )
        
        if min_height is not None and height < min_height:
            raise VideoValidationError(
                video_path, 
                f"Video height ({height}px) is smaller than minimum required ({min_height}px)"
            )
        
        # Check codec if specified
        if required_codecs:
            codec = video_stream.get('codec_name', '')
            if codec not in required_codecs:
                raise VideoValidationError(
                    video_path, 
                    f"Video codec '{codec}' is not in the list of required codecs: {required_codecs}"
                )
        
        # Check for corruption by attempting to read a frame
        if check_corruption:
            try:
                # Use ffmpeg to read the first frame without saving it
                corruption_cmd = [
                    'ffmpeg', '-v', 'error', '-i', str(video_path), 
                    '-f', 'null', '-frames:v', '1', '-'
                ]
                corruption_result = subprocess.run(
                    corruption_cmd, 
                    capture_output=True, 
                    text=True
                )
                
                if corruption_result.returncode != 0:
                    raise VideoValidationError(
                        video_path, 
                        f"Video file appears to be corrupted: {corruption_result.stderr}"
                    )
            except Exception as e:
                raise VideoValidationError(video_path, f"Error checking for corruption: {str(e)}")
        
        # Return video information
        return {
            'path': str(video_path),
            'duration': duration,
            'width': width,
            'height': height,
            'codec': video_stream.get('codec_name', ''),
            'format': info.get('format', {}).get('format_name', ''),
            'size': int(info.get('format', {}).get('size', 0)),
            'bitrate': int(info.get('format', {}).get('bit_rate', 0)),
            'fps': eval(video_stream.get('r_frame_rate', '0/1'))  # Convert fraction to float
        }
        
    except VideoValidationError:
        raise
    except Exception as e:
        raise VideoValidationError(video_path, f"Error validating video: {str(e)}") 