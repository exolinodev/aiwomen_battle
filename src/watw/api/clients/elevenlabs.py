"""
ElevenLabs API client for Women Around The World.

This module provides a client for interacting with the ElevenLabs API
for text-to-speech functionality.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from elevenlabs.client import ElevenLabs

from watw.api.clients.base import APIError, BaseAPIClient
from watw.api.config import Config


class ElevenLabsClient(BaseAPIClient):
    """
    Client for interacting with the ElevenLabs API.

    This client provides methods for generating voice-overs using
    the ElevenLabs text-to-speech API.
    """

    def __init__(self, require_api_key: bool = True):
        """
        Initialize the ElevenLabs API client.

        Args:
            require_api_key: Whether to require a valid API key
        """
        config = Config()
        api_key = config.get_setting("elevenlabs", "api_key", "")
        base_url = config.get_setting(
            "elevenlabs", "base_url", "https://api.elevenlabs.io"
        )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            service_name="elevenlabs",
            require_api_key=require_api_key,
        )

        self.config = config
        # Initialize the ElevenLabs client if API key is available
        self.client: Optional[ElevenLabs] = None
        if self.api_key:
            try:
                self.client = ElevenLabs(api_key=self.api_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize ElevenLabs client: {e}")
                # Continue without the client - methods will check self.client before use

        # Load default settings
        self.default_model = "eleven_multilingual_v2"
        self.default_output_format = "mp3_44100_128"

        # Mask API key for logging
        masked_key = (
            self.api_key[:8] + "*" * (len(self.api_key) - 8) if self.api_key else None
        )
        self.logger.info(f"Initialized ElevenLabs client with API key: {masked_key}")

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available voices.

        Returns:
            List of voice objects

        Raises:
            APIError: If client is not initialized or request fails
        """
        if not self.client:
            raise APIError("ElevenLabs client not initialized")

        try:
            voices_response = self.client.voices.get_all()  # Get response object
            voices = voices_response.voices  # Access the list of voices
            return [
                {
                    "voice_id": getattr(voice, "voice_id", None),
                    "name": getattr(voice, "name", "Unknown"),
                }
                for voice in voices
            ]
        except Exception as e:
            self.logger.error(f"Error getting available voices: {str(e)}")
            raise APIError(f"Failed to get available voices: {str(e)}")

    def get_voice_id(self, voice_name: Optional[str] = None) -> str:
        """
        Get the voice ID for a given voice name or use the first available voice.

        Args:
            voice_name: Optional name of the voice to use

        Returns:
            Voice ID

        Raises:
            APIError: If no voices are available or the specified voice is not found
        """
        if not self.client:
            raise APIError("ElevenLabs client not initialized")

        try:
            voices_response = self.client.voices.get_all()  # Get response object
            voices = voices_response.voices  # Access the list of voices
            if not voices:
                raise APIError("No voices available in your ElevenLabs account")

            if voice_name:
                for voice in voices:
                    if getattr(voice, "name", "").lower() == voice_name.lower():
                        voice_id = getattr(voice, "voice_id", "")
                        if voice_id:  # Ensure voice_id is not empty
                            return voice_id
                self.logger.warning(
                    f"Voice '{voice_name}' not found, using first available voice"
                )

            # If voice not found or not specified, use the first available voice
            first_voice_id = getattr(voices[0], "voice_id", "")
            if not first_voice_id:
                raise APIError("First available voice has no voice_id")
            return first_voice_id
        except Exception as e:
            self.logger.error(f"Error getting voice ID: {str(e)}")
            raise APIError(f"Failed to get voice ID: {str(e)}")

    def generate_speech(
        self,
        text: str,
        voice_name: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> bytes:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            voice_name: Name of the voice to use
            model_id: ID of the model to use
            output_format: Output format for the audio

        Returns:
            Audio data as bytes

        Raises:
            APIError: If speech generation fails
        """
        if not self.client:
            raise APIError("ElevenLabs client not initialized")

        try:
            # Get voice ID
            voice_id = self.get_voice_id(voice_name)

            # Use default model and format if not specified
            if model_id is None:
                model_id = self.default_model
            if output_format is None:
                output_format = self.default_output_format

            self.logger.info(
                f"Generating speech with voice: {voice_id}, model: {model_id}"
            )

            # Generate audio
            audio_generator = self.client.text_to_speech.convert(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                output_format=output_format,
            )

            # Convert generator to bytes
            audio_data = b"".join(audio_generator)

            return audio_data
        except Exception as e:
            self.logger.error(f"Error generating speech: {str(e)}")
            raise APIError(f"Failed to generate speech: {str(e)}")

    def generate_and_save(
        self,
        text: str,
        output_path: Union[str, Path],
        voice_name: Optional[str] = None,
        model_id: Optional[str] = None,
        output_format: Optional[str] = None,
    ) -> Path:
        """
        Generate speech from text and save it to a file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file
            voice_name: Name of the voice to use
            model_id: ID of the model to use
            output_format: Output format for the audio

        Returns:
            Path to the saved audio file

        Raises:
            APIError: If speech generation or saving fails
        """
        try:
            # Convert path to Path object
            output_path = Path(output_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate speech
            audio_data = self.generate_speech(
                text=text,
                voice_name=voice_name,
                model_id=model_id,
                output_format=output_format,
            )

            # Save to file
            with open(output_path, "wb") as f:
                f.write(audio_data)

            self.logger.info(f"Speech saved to {output_path}")

            return output_path
        except Exception as e:
            self.logger.error(f"Error generating and saving speech: {str(e)}")
            raise APIError(f"Failed to generate and save speech: {str(e)}")

    def generate_voiceover(
        self,
        script: str,
        output_path: Union[str, Path],
        voice_name: Optional[str] = None,
    ) -> Path:
        """
        Generate a voice-over from a script and save it to a file.

        Args:
            script: Text script for the voice-over
            output_path: Path to save the audio file
            voice_name: Name of the voice to use

        Returns:
            Path to the saved audio file

        Raises:
            APIError: If voice-over generation fails
        """
        return self.generate_and_save(
            text=script, output_path=output_path, voice_name=voice_name
        )
