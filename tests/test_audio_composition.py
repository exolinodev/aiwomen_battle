"""
Test audio composition functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from watw.api.clients.base import APIError
from watw.core.audio_composition import AudioComposition
from watw.core.voiceover import VoiceoverError, VoiceoverGenerator
from watw.utils.common import MockAPITestCase


class TestAudioComposition(MockAPITestCase):
    """Test class for audio composition functionality."""

    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.audio_composition = AudioComposition()

        # Load the existing mock voiceover file
        mock_voiceover_path = Path("tests/mockdata/voiceovers/voiceover_01_japan.mp3")
        with open(mock_voiceover_path, "rb") as f:
            self.mock_audio_data = f.read()
        self.mock_audio_generator = [self.mock_audio_data]

        # Create a mock ElevenLabs client
        self.mock_client = MagicMock()
        self.mock_client.generate_and_save.return_value = (
            self.output_dir / "test_voiceover.mp3"
        )

        # Patch the ElevenLabs client at the correct level
        self.elevenlabs_patcher = patch(
            "watw.core.voiceover.ElevenLabsClient", return_value=self.mock_client
        )
        self.elevenlabs_patcher.start()

        # Create the voiceover generator after patching
        self.voiceover_generator = VoiceoverGenerator()

    def tearDown(self):
        """Tear down the test case."""
        self.elevenlabs_patcher.stop()
        super().tearDown()

    def test_generate_voiceover_success(self):
        """Test successful voiceover generation."""
        # Generate voiceover
        output_path = self.output_dir / "test_voiceover.mp3"
        result_path = self.voiceover_generator.generate_voiceover(
            video_number=1,
            country1="Test Country 1",
            country2="Test Country 2",
            output_path=output_path,
            voice="test_voice",
            model="test_model",
        )

        # Assert the file exists and has content
        self.assert_file_exists(result_path)
        self.assert_file_size(result_path, min_size=len(self.mock_audio_data))

        # Verify the mock was called with correct parameters
        self.mock_client.generate_and_save.assert_called_once()
        call_args = self.mock_client.generate_and_save.call_args[1]
        self.assertEqual(call_args["voice_name"], "test_voice")
        self.assertEqual(call_args["model_id"], "test_model")

    def test_generate_voiceover_api_error(self):
        """Test voiceover generation with API error."""
        # Configure mock client to raise APIError
        self.mock_client.generate_and_save.side_effect = APIError("API error")

        # Attempt to generate voiceover
        output_path = self.output_dir / "test_voiceover.mp3"
        with self.assertRaises(VoiceoverError) as context:
            self.voiceover_generator.generate_voiceover(
                video_number=1,
                country1="Test Country 1",
                country2="Test Country 2",
                output_path=output_path,
                voice="test_voice",
                model="test_model",
            )

        # Verify the error message
        self.assertIn("API error", str(context.exception))

    def test_generate_voiceover_unexpected_error(self):
        """Test voiceover generation with unexpected error."""
        # Configure mock client to raise unexpected error
        self.mock_client.generate_and_save.side_effect = Exception("Unexpected error")

        # Attempt to generate voiceover
        output_path = self.output_dir / "test_voiceover.mp3"
        with self.assertRaises(VoiceoverError) as context:
            self.voiceover_generator.generate_voiceover(
                video_number=1,
                country1="Test Country 1",
                country2="Test Country 2",
                output_path=output_path,
                voice="test_voice",
                model="test_model",
            )

        # Verify the error message
        self.assertIn("Unexpected error", str(context.exception))

    def test_generate_voiceover_invalid_parameters(self):
        """Test voiceover generation with invalid parameters."""
        # Attempt to generate voiceover with invalid video number
        output_path = self.output_dir / "test_voiceover.mp3"
        with self.assertRaises(VoiceoverError) as context:
            self.voiceover_generator.generate_voiceover(
                video_number=0,  # Invalid video number
                country1="Test Country 1",
                country2="Test Country 2",
                output_path=output_path,
            )

        # Verify the error message
        self.assertIn("Invalid parameters", str(context.exception))
