"""
Video editing utilities for Women Around The World.

This module provides functions for editing and combining videos,
including adding voice-overs and background music.
"""

import os
from pathlib import Path
from typing import List, Optional, Union

from watw.utils.common.media_utils import (
    combine_video_with_audio,
    add_background_music,
    concatenate_videos,
    trim_video
)
from watw.utils.common.logging_utils import setup_logger, log_execution_time

# Set up logger
logger = setup_logger("watw.video_editor")

class VideoEditor:
    """Class for editing and combining videos."""
    
    def __init__(self, config=None):
        """Initialize the VideoEditor with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for video editing.
        """
        self.config = config or {}
        self.logger = setup_logger("watw.video_editor")
    
    @log_execution_time()
    def create_video(self, scenes, output_dir: Union[str, Path]) -> Path:
        """Create a video from a list of scenes.
        
        Args:
            scenes: List of scene descriptions
            output_dir: Directory to save the output
            
        Returns:
            Path to the final video
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # For now, just return a placeholder path
        # This will be implemented with actual video generation logic
        return output_dir / "final_video.mp4"

@log_execution_time()
def combine_video_with_voiceover(
    video_path: Union[str, Path],
    voiceover_path: Union[str, Path],
    output_path: Union[str, Path],
    background_music_path: Optional[Union[str, Path]] = None,
    is_first_clip: bool = False
) -> Path:
    """
    Combine a video with a voice-over and optional background music.
    For the first clip, allows the complete voice-over to play while keeping video at 3 seconds.
    For other clips, limits both video and voice-over to 3 seconds.
    
    Args:
        video_path: Path to the video file
        voiceover_path: Path to the voice-over audio file
        output_path: Path to save the final video
        background_music_path: Path to background music file (optional)
        is_first_clip: Whether this is the first clip (Japan)
        
    Returns:
        Path to the final video
    """
    # Convert paths to Path objects
    video_path = Path(video_path)
    voiceover_path = Path(voiceover_path)
    output_path = Path(output_path)
    
    # First, trim the video to 3 seconds
    temp_video = output_path.parent / f"{output_path.stem}_temp{output_path.suffix}"
    trim_video(video_path, temp_video, duration=3.0)
    
    # For the first clip, we want to keep the complete voice-over
    if is_first_clip:
        # Combine video with voice-over, padding audio to match video duration
        result_path = combine_video_with_audio(
            temp_video,
            voiceover_path,
            output_path,
            audio_pad=True,
            audio_pad_duration=3.0
        )
    else:
        # Combine video with voice-over, both limited to 3 seconds
        result_path = combine_video_with_audio(
            temp_video,
            voiceover_path,
            output_path
        )
    
    # Add background music if provided
    if background_music_path:
        result_path = add_background_music(
            result_path,
            background_music_path,
            output_path,
            music_volume=0.1
        )
    
    # Clean up temporary file
    if temp_video.exists():
        temp_video.unlink()
    
    return result_path

@log_execution_time()
def create_final_video(
    video_clips_dir: Union[str, Path],
    voiceover_path: Union[str, Path],
    output_dir: Union[str, Path],
    background_music_path: Optional[Union[str, Path]] = None,
    music_volume: float = 0.1
) -> Path:
    """
    Create a final video by combining multiple video clips with a voice-over
    and optional background music.
    
    Args:
        video_clips_dir: Directory containing video clips
        voiceover_path: Path to the voice-over audio file
        output_dir: Directory to save the final video
        background_music_path: Path to background music file (optional)
        music_volume: Volume of the background music (0.0 to 1.0)
        
    Returns:
        Path to the final video
    """
    # Convert paths to Path objects
    video_clips_dir = Path(video_clips_dir)
    voiceover_path = Path(voiceover_path)
    output_dir = Path(output_dir)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of video clips
    video_clips = sorted([f for f in video_clips_dir.glob("*.mp4")])
    
    if not video_clips:
        raise ValueError(f"No video clips found in {video_clips_dir}")
    
    # Process each video clip
    processed_clips = []
    for i, clip_path in enumerate(video_clips):
        # Create output path for this clip
        clip_output_path = output_dir / f"processed_{clip_path.name}"
        
        # Combine video with voice-over
        is_first_clip = (i == 0)  # First clip is Japan
        combine_video_with_voiceover(
            clip_path,
            voiceover_path,
            clip_output_path,
            background_music_path=None,  # We'll add background music at the end
            is_first_clip=is_first_clip
        )
        
        processed_clips.append(clip_output_path)
    
    # Concatenate all processed clips
    final_output_path = output_dir / "final_video.mp4"
    concatenate_videos(processed_clips, final_output_path)
    
    # Add background music to the final video
    if background_music_path:
        final_output_path = add_background_music(
            final_output_path,
            background_music_path,
            output_dir / "final_video_with_music.mp4",
            music_volume=music_volume
        )
    
    return final_output_path

@log_execution_time()
def concatenate_videos(concat_list_path: Union[str, Path], output_path: Union[str, Path], working_dir: Optional[Union[str, Path]] = None) -> Path:
    """
    Concatenate videos listed in a file.
    
    Args:
        concat_list_path: Path to the file containing the list of videos to concatenate
        output_path: Path to save the concatenated video
        working_dir: Working directory for temporary files
        
    Returns:
        Path to the concatenated video
    """
    # Convert paths to Path objects
    concat_list_path = Path(concat_list_path)
    output_path = Path(output_path)
    
    # Ensure output directory exists
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create working directory if not provided
    if working_dir is None:
        working_dir = output_dir / "temp"
    else:
        working_dir = Path(working_dir)
    
    working_dir.mkdir(parents=True, exist_ok=True)
    
    # Concatenate videos
    result_path = concatenate_videos(
        [line.strip() for line in open(concat_list_path, 'r').readlines()],
        output_path,
        working_dir
    )
    
    return result_path 