"""
Voice-over generation utilities for Women Around The World.

This module provides functions for generating voice-overs for videos
using the ElevenLabs API.
"""

import os
from pathlib import Path
from typing import Optional, Union

from elevenlabs.client import ElevenLabs

from watw.utils.common.api_utils import api_config
from watw.utils.common.logging_utils import setup_logger, log_execution_time

# Set up logger
logger = setup_logger("watw.voiceover")

class VoiceoverGenerator:
    """
    Class for generating voice-overs using ElevenLabs API.
    """
    def __init__(self, config=None):
        """
        Initialize the VoiceoverGenerator.
        
        Args:
            config: Optional configuration object. If None, uses default configuration.
        """
        self.config = config or api_config
        self.api_key = self.config.get_api_key("elevenlabs")
        self.client = ElevenLabs(api_key=self.api_key)
        
        # Log API key (masked)
        masked_key = self.api_key[:8] + "*" * (len(self.api_key) - 8) if self.api_key else None
        logger.info(f"Using ElevenLabs API key: {masked_key}")
    
    @log_execution_time()
    def generate_voiceover(
        self,
        video_number: int,
        country1: str,
        country2: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        model: Optional[str] = None
    ) -> Path:
        """
        Generate a voice-over for a video with the specified parameters.
        
        Args:
            video_number: The video number (1-100)
            country1: The first country name
            country2: The second country name
            output_path: Path to save the audio file
            voice: The voice to use (default: from config)
            model: The model to use (default: from config)
            
        Returns:
            Path to the generated audio file
        """
        # Convert path to Path object
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get voice and model from config if not provided
        if voice is None:
            voice = self.config.config_data["elevenlabs"]["voice"]
        if model is None:
            model = self.config.config_data["elevenlabs"]["model"]
        
        # Create the script
        script = f"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. "
        script += "Don't forget to like and subscribe to our channel for more amazing content!"
        
        logger.info(f"Generating voice-over for video #{video_number}")
        
        # Generate audio using the new API
        audio_generator = self.client.text_to_speech.convert(
            text=script,
            voice_id=voice,
            model_id=model,
            output_format="mp3_44100_128"
        )
        
        # Convert generator to bytes
        audio_data = b''.join(audio_generator)
        
        # Save the audio to a file
        with open(output_path, 'wb') as f:
            f.write(audio_data)
        
        logger.info(f"Voice-over saved to {output_path}")
        
        return output_path
    
    @log_execution_time()
    def generate_voiceover_for_video(
        self,
        video_number: int,
        country1: str,
        country2: str,
        output_dir: Union[str, Path]
    ) -> Path:
        """
        Generate a voice-over for a video and save it to the specified directory.
        
        Args:
            video_number: The video number (1-100)
            country1: The first country name
            country2: The second country name
            output_dir: Directory to save the audio file
            
        Returns:
            Path to the generated audio file
        """
        # Convert path to Path object
        output_dir = Path(output_dir)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output path
        output_path = output_dir / f"voiceover_{video_number}.mp3"
        
        # Generate voice-over
        return self.generate_voiceover(
            video_number=video_number,
            country1=country1,
            country2=country2,
            output_path=output_path
        )

# Create a default instance for backward compatibility
voiceover_generator = VoiceoverGenerator()
generate_voiceover = voiceover_generator.generate_voiceover
generate_voiceover_for_video = voiceover_generator.generate_voiceover_for_video

# For backward compatibility with the old module-level functions
__all__ = ['VoiceoverGenerator', 'generate_voiceover', 'generate_voiceover_for_video'] 