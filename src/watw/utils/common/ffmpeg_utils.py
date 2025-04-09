"""
FFmpeg command construction utilities.

This module provides helper functions for constructing FFmpeg commands
for various video and audio processing operations.
"""

from typing import List, Union, Optional, Dict, Any
from pathlib import Path
import subprocess
from .validation_utils import validate_file_exists
from .exceptions import FFmpegError

def build_trim_command(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    duration: float = 3.0,
    start_time: float = 0.0
) -> List[str]:
    """
    Build FFmpeg command for trimming a video.
    
    Args:
        input_path: Path to input video
        output_path: Path to save trimmed video
        duration: Duration in seconds
        start_time: Start time in seconds
        
    Returns:
        List of command arguments
    """
    return [
        'ffmpeg', '-y',
        '-i', str(input_path),
        '-ss', str(start_time),
        '-t', str(duration),
        '-c:v', 'copy',
        str(output_path)
    ]

def build_combine_video_audio_command(
    video_path: Union[str, Path],
    audio_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_pad: bool = False,
    audio_pad_duration: Optional[float] = None
) -> List[str]:
    """
    Build FFmpeg command for combining video with audio.
    
    Args:
        video_path: Path to video file
        audio_path: Path to audio file
        output_path: Path to save combined file
        audio_pad: Whether to pad audio to match video duration
        audio_pad_duration: Duration to pad audio to (if None, uses video duration)
        
    Returns:
        List of command arguments
    """
    if audio_pad:
        # Use filter_complex to pad audio
        return [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-filter_complex', f'[1:a]apad=whole_dur={audio_pad_duration}[a1]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v',
            '-map', '[a1]',
            str(output_path)
        ]
    else:
        # Simple combination without padding
        return [
            'ffmpeg', '-y',
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',
            '-c:a', 'aac',
            str(output_path)
        ]

def build_add_background_music_command(
    video_path: Union[str, Path],
    music_path: Union[str, Path],
    output_path: Union[str, Path],
    music_volume: float = 0.1
) -> List[str]:
    """
    Build FFmpeg command for adding background music to a video.
    
    Args:
        video_path: Path to video file
        music_path: Path to background music file
        output_path: Path to save video with background music
        music_volume: Volume of background music (0.0 to 1.0)
        
    Returns:
        List of command arguments
    """
    return [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(music_path),
        '-filter_complex', f'[1:a]volume={music_volume}[a1];[0:a][a1]amix=inputs=2:duration=first',
        '-c:v', 'copy',
        str(output_path)
    ]

def build_concatenate_videos_command(
    concat_list_path: Union[str, Path],
    output_path: Union[str, Path],
    reencode: bool = False,
    video_codec: str = 'libx264',
    preset: str = 'medium',
    crf: int = 22
) -> List[str]:
    """
    Build FFmpeg command for concatenating videos.
    
    Args:
        concat_list_path: Path to file listing videos to concatenate
        output_path: Path to save concatenated video
        reencode: Whether to re-encode the video
        video_codec: Video codec to use if reencoding
        preset: Encoding preset to use if reencoding
        crf: Constant Rate Factor to use if reencoding
        
    Returns:
        List of command arguments
    """
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', str(concat_list_path)
    ]
    
    if reencode:
        cmd.extend([
            '-c:v', video_codec,
            '-preset', preset,
            '-crf', str(crf)
        ])
    else:
        cmd.extend(['-c', 'copy'])
    
    cmd.append(str(output_path))
    return cmd

def build_extract_audio_command(
    video_path: Union[str, Path],
    output_path: Union[str, Path],
    format: str = "mp3"
) -> List[str]:
    """
    Build FFmpeg command for extracting audio from a video.
    
    Args:
        video_path: Path to video file
        output_path: Path to save extracted audio
        format: Audio format (mp3, wav, etc.)
        
    Returns:
        List of command arguments
    """
    return [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-vn',  # No video
        '-acodec', 'libmp3lame' if format == "mp3" else 'pcm_s16le',
        str(output_path)
    ]

def build_get_video_info_command(video_path: Union[str, Path]) -> List[str]:
    """
    Build FFprobe command for getting video information.
    
    Args:
        video_path: Path to video file
        
    Returns:
        List of command arguments
    """
    return [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration,size,bit_rate',
        '-show_entries', 'stream=width,height,codec_name,codec_type',
        '-of', 'json',
        str(video_path)
    ]

def build_ensure_shorts_resolution_command(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    target_width: int = 1080,
    target_height: int = 1920,
    video_codec: str = 'libx264',
    preset: str = 'medium',
    bitrate: str = '4M',
    fps: int = 30,
    pix_fmt: str = 'yuv420p'
) -> List[str]:
    """
    Build FFmpeg command for ensuring video is in Shorts resolution (9:16).
    
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
        'ffmpeg', '-i', str(input_path),
        '-vf', f'scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,'
               f'pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2:color=black',
        '-c:v', video_codec,
        '-preset', preset,
        '-b:v', bitrate,
        '-maxrate', bitrate,
        '-bufsize', '2M',
        '-pix_fmt', pix_fmt,
        '-r', str(fps),
        '-c:a', 'copy',
        str(output_path),
        '-y'
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
    video_codec: str = 'libx264',
    preset: str = 'medium',
    bitrate: str = '4M',
    audio_codec: str = 'aac',
    audio_bitrate: str = '192k',
    fps: int = 30,
    pix_fmt: str = 'yuv420p'
) -> List[str]:
    """
    Build FFmpeg command for final video muxing with audio.
    
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
        'ffmpeg', '-loglevel', 'warning',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-map', '0:v:0',
        '-map', '1:a:0?'
    ]
    
    if duration is not None:
        cmd.extend(['-t', f'{duration:.3f}'])
    
    if video_fade and video_filter:
        cmd.extend(['-vf', video_filter])
        cmd.extend([
            '-c:v', video_codec,
            '-preset', preset,
            '-b:v', bitrate,
            '-maxrate', bitrate,
            '-bufsize', '2M',
            '-pix_fmt', pix_fmt,
            '-r', str(fps)
        ])
    else:
        cmd.extend(['-c:v', 'copy'])
    
    cmd.extend([
        '-c:a', audio_codec,
        '-b:a', audio_bitrate,
        '-ac', '2',
        '-ar', '48000'
    ])
    
    if audio_fade and audio_filter:
        cmd.extend(['-af', audio_filter])
    
    cmd.extend([str(output_path), '-y'])
    return cmd 