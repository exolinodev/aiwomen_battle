"""
Audio composition utilities for Women Around The World.

This module provides functions for composing audio, including
combining voice-overs with background music.
"""

import os
from pathlib import Path
from typing import Optional, Union

from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.audio.AudioClip import CompositeAudioClip

from watw.utils.common.logging_utils import setup_logger, log_execution_time

# Set up logger
logger = setup_logger("watw.audio")

class AudioComposition:
    """Class for composing audio from multiple sources."""
    
    def __init__(self, config=None):
        """Initialize the AudioComposition with optional configuration.
        
        Args:
            config (dict, optional): Configuration parameters for audio composition.
        """
        self.config = config or {}
        self.logger = setup_logger("watw.audio")
    
    @log_execution_time()
    def combine_audio(
        self,
        voiceover_path: Union[str, Path],
        background_music_path: Union[str, Path],
        output_path: Union[str, Path],
        music_volume: float = 0.1
    ) -> Path:
        """
        Combine voice-over with background music.
        
        Args:
            voiceover_path: Path to the voice-over audio file
            background_music_path: Path to the background music file
            output_path: Path to save the combined audio
            music_volume: Volume of the background music (0.0 to 1.0)
            
        Returns:
            Path to the combined audio file
        """
        # Convert paths to Path objects
        voiceover_path = Path(voiceover_path)
        background_music_path = Path(background_music_path)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load audio clips
        voiceover = AudioFileClip(str(voiceover_path))
        background_music = AudioFileClip(str(background_music_path))
        
        # Loop background music if needed
        if background_music.duration < voiceover.duration:
            background_music = background_music.loop(duration=voiceover.duration)
        else:
            background_music = background_music.subclip(0, voiceover.duration)
        
        # Set background music volume
        background_music = background_music.volumex(music_volume)
        
        # Combine audio clips
        combined = CompositeAudioClip([voiceover, background_music])
        
        # Write to file
        combined.write_audiofile(str(output_path))
        
        # Clean up
        voiceover.close()
        background_music.close()
        combined.close()
        
        return output_path
    
    @log_execution_time()
    def trim_audio(
        self,
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        start_time: float = 0.0,
        end_time: Optional[float] = None
    ) -> Path:
        """
        Trim an audio file to the specified duration.
        
        Args:
            audio_path: Path to the audio file
            output_path: Path to save the trimmed audio
            start_time: Start time in seconds
            end_time: End time in seconds (optional)
            
        Returns:
            Path to the trimmed audio file
        """
        # Convert paths to Path objects
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load audio clip
        audio = AudioFileClip(str(audio_path))
        
        # Trim audio
        if end_time is None:
            trimmed = audio.subclip(start_time)
        else:
            trimmed = audio.subclip(start_time, end_time)
        
        # Write to file
        trimmed.write_audiofile(str(output_path))
        
        # Clean up
        audio.close()
        trimmed.close()
        
        return output_path

# Create a default instance for backward compatibility
audio_composition = AudioComposition()
combine_audio = audio_composition.combine_audio
trim_audio = audio_composition.trim_audio

# For backward compatibility with the old module-level functions
__all__ = ['AudioComposition', 'combine_audio', 'trim_audio'] 