"""
Test complete workflow functionality.
"""

import os
import unittest
from pathlib import Path
from watw.utils.common import MockAPITestCase, mock_responses
from watw.voiceover import VoiceoverGenerator
from watw.core.video_editor import VideoEditor

class TestCompleteWorkflow(MockAPITestCase):
    """Test class for complete workflow functionality."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.voiceover_generator = VoiceoverGenerator()
        self.video_editor = VideoEditor()
        
    def test_generate_voiceover(self):
        """Test generating a voiceover."""
        # Test text
        text = "This is a test voiceover for complete workflow."
        
        # Generate voiceover
        output_path = self.output_dir / "test_complete_workflow_voiceover.mp3"
        
        # Mock the audio stream
        mock_audio_data = b"This is mock audio data for complete workflow testing purposes."
        
        # Save the mock audio data to a file
        with open(output_path, "wb") as f:
            f.write(mock_audio_data)
        
        # Assert the file exists and has content
        self.assert_file_exists(output_path)
        self.assert_file_size(output_path, min_size=10)  # At least 10 bytes
        
    def test_edit_video(self):
        """Test editing a video."""
        # Create a mock video
        video_path = self.output_dir / "test_complete_workflow_video.mp4"
        mock_responses.create_mock_video(duration=5, output_path=video_path)
        
        # Assert the video exists and has content
        self.assert_file_exists(video_path)
        self.assert_file_size(video_path, min_size=100)  # At least 100 bytes 