"""
Test complete workflow functionality.
"""

from unittest.mock import MagicMock, patch

from watw.api.clients.base import APIError
from watw.core.video.base import VideoEditor
from watw.core.voiceover import VoiceoverError, VoiceoverGenerator
from watw.utils.common import MockAPITestCase, mock_responses


class TestCompleteWorkflow(MockAPITestCase):
    """Test class for complete workflow functionality."""

    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.mock_client = MagicMock()
        self.patcher = patch(
            "watw.core.voiceover.ElevenLabsClient", return_value=self.mock_client
        )
        self.patcher.start()
        self.voiceover_generator = VoiceoverGenerator()
        self.video_editor = VideoEditor()

    def test_generate_voiceover_success(self):
        """Test successful voiceover generation."""
        # Test text
        text = "This is a test voiceover for complete workflow."

        # Generate voiceover
        output_path = self.output_dir / "test_complete_workflow_voiceover.mp3"

        # Mock the audio stream
        mock_audio_data = (
            b"This is mock audio data for complete workflow testing purposes."
        )

        # Configure mock client
        self.mock_client.generate_and_save.return_value = output_path

        # Save the mock audio data to a file
        with open(output_path, "wb") as f:
            f.write(mock_audio_data)

        # Generate voiceover
        result_path = self.voiceover_generator.generate(
            script=text, output_path=output_path
        )

        # Assert the file exists and has content
        self.assert_file_exists(result_path)
        self.assert_file_size(result_path, min_size=10)  # At least 10 bytes

        # Verify the client was called correctly
        self.mock_client.generate_and_save.assert_called_once()

    def test_generate_voiceover_api_error(self):
        """Test voiceover generation with API error."""
        # Test text
        text = "This is a test voiceover for complete workflow."
        output_path = self.output_dir / "test_complete_workflow_voiceover.mp3"

        # Configure mock client to raise APIError
        self.mock_client.generate_and_save.side_effect = APIError("API error")

        # Attempt to generate voiceover
        with self.assertRaises(VoiceoverError) as context:
            self.voiceover_generator.generate(script=text, output_path=output_path)

        # Verify the error message
        self.assertIn("API error", str(context.exception))

    def test_generate_voiceover_unexpected_error(self):
        """Test voiceover generation with unexpected error."""
        # Test text
        text = "This is a test voiceover for complete workflow."
        output_path = self.output_dir / "test_complete_workflow_voiceover.mp3"

        # Configure mock client to raise unexpected error
        self.mock_client.generate_and_save.side_effect = Exception("Unexpected error")

        # Attempt to generate voiceover
        with self.assertRaises(VoiceoverError) as context:
            self.voiceover_generator.generate(script=text, output_path=output_path)

        # Verify the error message
        self.assertIn("Unexpected error", str(context.exception))

    def test_edit_video(self):
        """Test editing a video."""
        # Create a mock video
        video_path = self.output_dir / "test_complete_workflow_video.mp4"
        mock_responses.create_mock_video(duration=5, output_path=video_path)

        # Assert the video exists and has content
        self.assert_file_exists(video_path)
        self.assert_file_size(video_path, min_size=100)  # At least 100 bytes

    def tearDown(self):
        """Clean up after tests."""
        self.patcher.stop()
        super().tearDown()
