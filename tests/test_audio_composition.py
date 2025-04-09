"""
Test audio composition functionality.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
from watw.utils.common import MockAPITestCase
from watw.core.audio_composition import AudioComposition
from watw.core.voiceover import VoiceoverGenerator

class TestAudioComposition(MockAPITestCase):
    """Test class for audio composition functionality."""
    
    def setUp(self):
        """Set up the test case."""
        super().setUp()
        self.audio_composition = AudioComposition()
        
        # Load the existing mock voiceover file
        mock_voiceover_path = Path("tests/mockdata/voiceovers/voiceover_01_japan.mp3")
        with open(mock_voiceover_path, 'rb') as f:
            self.mock_audio_data = f.read()
        self.mock_audio_generator = [self.mock_audio_data]
        
        # Create a mock ElevenLabs client
        self.mock_client = MagicMock()
        self.mock_client.text_to_speech.convert.return_value = self.mock_audio_generator
        
        # Patch the ElevenLabs client at the correct level
        self.elevenlabs_patcher = patch('watw.core.voiceover.ElevenLabs', return_value=self.mock_client)
        self.elevenlabs_patcher.start()
        
        # Create the voiceover generator after patching
        self.voiceover_generator = VoiceoverGenerator()
        
    def tearDown(self):
        """Tear down the test case."""
        self.elevenlabs_patcher.stop()
        super().tearDown()
        
    def test_generate_voiceover(self):
        """Test generating a voiceover."""
        # Generate voiceover
        output_path = self.output_dir / "test_voiceover.mp3"
        self.voiceover_generator.generate_voiceover(
            video_number=1,
            country1="Test Country 1",
            country2="Test Country 2",
            output_path=output_path,
            voice="test_voice",
            model="test_model"
        )
        
        # Assert the file exists and has content
        self.assert_file_exists(output_path)
        self.assert_file_size(output_path, min_size=len(self.mock_audio_data))
        
        # Verify the mock was called with correct parameters
        self.mock_client.text_to_speech.convert.assert_called_once()
        call_args = self.mock_client.text_to_speech.convert.call_args[1]
        self.assertEqual(call_args['voice_id'], 'test_voice')
        self.assertEqual(call_args['model_id'], 'test_model')
        self.assertEqual(call_args['output_format'], 'mp3_44100_128') 