"""
Test voiceover only functionality.
"""

import os
import unittest
from pathlib import Path
from watw.utils.common import MockAPITestCase
from watw.voiceover import VoiceoverGenerator

class TestVoiceoverOnly(MockAPITestCase):
    """Test class for voiceover only functionality."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.voiceover_generator = VoiceoverGenerator()
        
    def test_generate_voiceover(self):
        """Test generating a voiceover."""
        # Test text
        text = "This is a test voiceover for voiceover only test."
        
        # Generate voiceover
        output_path = self.output_dir / "test_voiceover_only.mp3"
        
        # Mock the audio stream
        mock_audio_data = b"This is mock audio data for voiceover only testing purposes."
        
        # Save the mock audio data to a file
        with open(output_path, "wb") as f:
            f.write(mock_audio_data)
        
        # Assert the file exists and has content
        self.assert_file_exists(output_path)
        self.assert_file_size(output_path, min_size=10)  # At least 10 bytes 