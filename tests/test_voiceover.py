"""
Test voiceover functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tests.test_utils import create_temp_directory
from watw.api.clients.base import APIError
from watw.core.voiceover import VoiceoverError, VoiceoverGenerator


@pytest.fixture
def mock_client():
    """Create a mock ElevenLabsClient."""
    client = MagicMock()
    client.generate_and_save.return_value = Path("path/to/mock/output.mp3")
    return client


@pytest.fixture
def generator(mock_client):
    """Create a VoiceoverGenerator with mocked client."""
    with patch("watw.core.voiceover.ElevenLabsClient", return_value=mock_client):
        return VoiceoverGenerator()


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    with create_temp_directory() as temp_dir:
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(exist_ok=True)
        yield output_dir


def test_generate(generator, mock_client, output_dir):
    """Test the core generate method."""
    # Test parameters
    script = "This is a test script."
    output_path = output_dir / "test_voiceover.mp3"
    voice_name = "test_voice"
    model_id = "test_model"

    # Call the generate method
    result = generator.generate(
        script=script, output_path=output_path, voice_name=voice_name, model_id=model_id
    )

    # Assert the result is a Path
    assert isinstance(result, Path)

    # Verify the mock was called with correct parameters
    mock_client.generate_and_save.assert_called_once_with(
        text=script, output_path=output_path, voice_name=voice_name, model_id=model_id
    )


def test_generate_voiceover(generator, mock_client, output_dir):
    """Test generating a voiceover with specific video parameters."""
    # Test parameters
    video_number = 1
    country1 = "Japan"
    country2 = "France"
    output_path = output_dir / "test_voiceover.mp3"
    voice = "test_voice"
    model = "test_model"

    # Call the generate_voiceover method
    result = generator.generate_voiceover(
        video_number=video_number,
        country1=country1,
        country2=country2,
        output_path=output_path,
        voice=voice,
        model=model,
    )

    # Assert the result is a Path
    assert isinstance(result, Path)

    # Verify the mock was called with correct parameters
    mock_client.generate_and_save.assert_called_once()
    call_args = mock_client.generate_and_save.call_args[1]
    assert call_args["voice_name"] == voice
    assert call_args["model_id"] == model

    # Verify the script contains the expected text
    script = call_args["text"]
    assert f"Video #{video_number}" in script
    assert country1 in script
    assert country2 in script


def test_generate_voiceover_for_video(generator, mock_client, output_dir):
    """Test generating a voiceover for a video with directory output."""
    # Test parameters
    video_number = 1
    country1 = "Japan"
    country2 = "France"

    # Call the generate_voiceover_for_video method
    result = generator.generate_voiceover_for_video(
        video_number=video_number,
        country1=country1,
        country2=country2,
        output_dir=output_dir,
    )

    # Assert the result is a Path
    assert isinstance(result, Path)

    # Verify the output path is in the correct directory
    assert result.parent == output_dir
    assert result.name == f"voiceover_{video_number}.mp3"

    # Verify the mock was called
    mock_client.generate_and_save.assert_called_once()


def test_client_initialization_failure():
    """Test VoiceoverGenerator initialization failure."""
    # Configure the mock to raise an APIError
    with patch(
        "watw.core.voiceover.ElevenLabsClient",
        side_effect=APIError("API key not found"),
    ):
        with pytest.raises(VoiceoverError) as exc_info:
            VoiceoverGenerator()
        assert "Failed to initialize ElevenLabs client" in str(exc_info.value)


def test_generate_api_error(generator, mock_client, output_dir):
    """Test generate method with API error."""
    # Configure the mock to raise an APIError
    mock_client.generate_and_save.side_effect = APIError("API error")

    with pytest.raises(VoiceoverError) as exc_info:
        generator.generate(script="test", output_path=output_dir / "test.mp3")
    assert "API error generating voice-over" in str(exc_info.value)


def test_generate_unexpected_error(generator, mock_client, output_dir):
    """Test generate method with unexpected error."""
    # Configure the mock to raise an unexpected error
    mock_client.generate_and_save.side_effect = IOError("File system error")

    with pytest.raises(VoiceoverError) as exc_info:
        generator.generate(script="test", output_path=output_dir / "test.mp3")
    assert "Unexpected error generating voice-over" in str(exc_info.value)


def test_generate_voiceover_api_error(generator, mock_client, output_dir):
    """Test generate_voiceover method with API error."""
    # Configure the mock to raise an APIError
    mock_client.generate_and_save.side_effect = APIError("API error")

    with pytest.raises(VoiceoverError) as exc_info:
        generator.generate_voiceover(
            video_number=1,
            country1="Japan",
            country2="France",
            output_path=output_dir / "test.mp3",
        )
    assert "API error generating voice-over" in str(exc_info.value)


def test_generate_voiceover_unexpected_error(generator, mock_client, output_dir):
    """Test generate_voiceover method with unexpected error."""
    # Configure the mock to raise an unexpected error
    mock_client.generate_and_save.side_effect = IOError("File system error")

    with pytest.raises(VoiceoverError) as exc_info:
        generator.generate_voiceover(
            video_number=1,
            country1="Japan",
            country2="France",
            output_path=output_dir / "test.mp3",
        )
    assert "Unexpected error generating voice-over" in str(exc_info.value)
