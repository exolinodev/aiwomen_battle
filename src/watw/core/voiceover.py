"""
Voice-over generation utilities for Women Around The World.

This module provides functions for generating voice-overs for videos
using the ElevenLabs API. It handles API communication, error management,
and file operations in a robust manner.

Example:
    ```python
    from watw.core.voiceover import VoiceoverGenerator, VoiceoverError

    try:
        generator = VoiceoverGenerator()
        output_path = generator.generate_voiceover(
            video_number=1,
            country1="Japan",
            country2="France",
            output_path="output/voiceover.mp3"
        )
        print(f"Generated voice-over at: {output_path}")
    except VoiceoverError as e:
        print(f"Failed to generate voice-over: {e}")
    ```
"""

from pathlib import Path
from typing import Optional, Union

from watw.api.clients.base import APIError
from watw.api.clients.elevenlabs import ElevenLabsClient
from watw.utils.common.logging_utils import log_execution_time, setup_logger

# Set up logger
logger = setup_logger("watw.voiceover")


class VoiceoverError(Exception):
    """Exception raised for voice-over generation errors.

    This exception wraps various types of errors that can occur during
    voice-over generation, including:
    - API initialization failures
    - API communication errors
    - File system errors
    - Unexpected errors

    Example:
        ```python
        try:
            generator = VoiceoverGenerator()
            generator.generate_voiceover(...)
        except VoiceoverError as e:
            print(f"Voice-over generation failed: {e}")
        ```
    """

    pass


class VoiceoverGenerator:
    """
    Class for generating voice-overs using ElevenLabs API.

    This class provides a high-level interface for generating voice-overs
    with proper error handling and logging. It manages the ElevenLabs client
    lifecycle and provides methods for different voice-over generation scenarios.

    Attributes:
        client (ElevenLabsClient): The underlying API client for voice generation
    """

    def __init__(self) -> None:
        """Initialize the VoiceoverGenerator.

        This method creates and initializes the ElevenLabs client. It handles
        potential initialization errors and wraps them in VoiceoverError.

        Raises:
            VoiceoverError: If client initialization fails due to:
                - Missing or invalid API key
                - Network connectivity issues
                - Unexpected errors during initialization
        """
        try:
            self.client = ElevenLabsClient()
            logger.info("Initialized VoiceoverGenerator with ElevenLabs client")
        except APIError as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            raise VoiceoverError(f"Failed to initialize ElevenLabs client: {e}")
        except Exception as e:
            logger.error(f"Unexpected error initializing VoiceoverGenerator: {e}")
            raise VoiceoverError(
                f"Unexpected error initializing VoiceoverGenerator: {e}"
            )

    @log_execution_time(logger)
    def generate_voiceover(
        self,
        video_number: int,
        country1: str,
        country2: str,
        output_path: Union[str, Path],
        voice: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Path:
        """
        Generate a voice-over for a video with the specified parameters.

        This method creates a script based on the video number and countries,
        then uses the core generate method to create the voice-over.

        Args:
            video_number: The video number (1-100)
            country1: The first country name
            country2: The second country name
            output_path: Path to save the audio file
            voice: The voice to use (default: first available voice)
            model: The model to use (default: from config)

        Returns:
            Path to the generated audio file

        Raises:
            VoiceoverError: If voice-over generation fails due to:
                - API communication errors
                - File system errors
                - Invalid parameters
                - Unexpected errors

        Example:
            ```python
            generator = VoiceoverGenerator()
            output_path = generator.generate_voiceover(
                video_number=1,
                country1="Japan",
                country2="France",
                output_path="output/voiceover.mp3",
                voice="Rachel",
                model="eleven_monolingual_v1"
            )
            ```
        """
        # Create the script
        script = f"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. "
        script += "Don't forget to like and subscribe to our channel for more amazing content!"

        logger.info(f"Generating voice-over for video #{video_number}")

        # Generate voice-over using the core generate method
        return self.generate(
            script=script, output_path=output_path, voice_name=voice, model_id=model
        )

    @log_execution_time(logger)
    def generate_voiceover_for_video(
        self,
        video_number: int,
        country1: str,
        country2: str,
        output_dir: Union[str, Path],
    ) -> Path:
        """
        Generate a voice-over for a video and save it to the specified directory.

        This method creates a standardized filename based on the video number
        and ensures the output directory exists before generating the voice-over.

        Args:
            video_number: The video number (1-100)
            country1: The first country name
            country2: The second country name
            output_dir: Directory to save the audio file

        Returns:
            Path to the generated audio file

        Raises:
            VoiceoverError: If voice-over generation fails due to:
                - API communication errors
                - File system errors
                - Invalid parameters
                - Unexpected errors

        Example:
            ```python
            generator = VoiceoverGenerator()
            output_path = generator.generate_voiceover_for_video(
                video_number=1,
                country1="Japan",
                country2="France",
                output_dir="output/voiceovers"
            )
            ```
        """
        # Convert path to Path object
        output_dir = Path(output_dir)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create output path
        output_path = output_dir / f"voiceover_{video_number}.mp3"

        # Generate voice-over using the core generate method
        return self.generate_voiceover(
            video_number=video_number,
            country1=country1,
            country2=country2,
            output_path=output_path,
        )

    @log_execution_time(logger)
    def generate(
        self,
        script: str,
        output_path: Union[str, Path],
        voice_name: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Path:
        """
        Generate a voice-over from a script and save it to a file.

        This is the core method that handles the actual voice-over generation
        using the ElevenLabs API. It manages file operations and error handling.

        Args:
            script: Text script for the voice-over
            output_path: Path to save the audio file
            voice_name: Name of the voice to use
            model_id: ID of the model to use

        Returns:
            Path to the saved audio file

        Raises:
            VoiceoverError: If voice-over generation fails due to:
                - Client not initialized
                - API communication errors
                - File system errors
                - Invalid parameters
                - Unexpected errors

        Example:
            ```python
            generator = VoiceoverGenerator()
            output_path = generator.generate(
                script="Hello, this is a test voice-over.",
                output_path="output/test.mp3",
                voice_name="Rachel",
                model_id="eleven_monolingual_v1"
            )
            ```
        """
        if not self.client:
            raise VoiceoverError("Client not initialized")

        # Convert path to Path object
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Generating voice-over with voice: {voice_name or 'default'}")
            return self.client.generate_and_save(
                text=script,
                output_path=output_path,
                voice_name=voice_name,
                model_id=model_id,
            )
        except APIError as e:
            logger.error(f"API error generating voice-over: {e}")
            raise VoiceoverError(f"API error generating voice-over: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating voice-over: {e}")
            raise VoiceoverError(f"Unexpected error generating voice-over: {e}")


# Remove module-level instance and function assignments
__all__ = ["VoiceoverGenerator", "VoiceoverError"]
