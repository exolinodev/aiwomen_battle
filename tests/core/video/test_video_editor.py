#!/usr/bin/env python3
import os

import pytest

from watw.core.video.video_editor import combine_video_with_voiceover


@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary test directory."""
    test_dir = tmp_path / "test_video_editor"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture
def test_video_path(test_dir):
    """Create or get test video path."""
    video_path = test_dir / "test_video.mp4"
    if not video_path.exists():
        # Create a simple video using FFmpeg
        os.system(
            f"ffmpeg -y -f lavfi -i color=c=blue:s=1280x720:d=5 -c:v libx264 {video_path}"
        )
    return video_path


@pytest.fixture
def test_voiceover_path(test_dir):
    """Get test voiceover path."""
    voiceover_path = test_dir / "voiceover_1.mp3"
    if not voiceover_path.exists():
        pytest.skip(
            "Voice-over file not found. Please run test_voiceover_only.py first."
        )
    return voiceover_path


def test_video_editor(test_dir, test_video_path, test_voiceover_path):
    """Test the video editor to combine a video with a voice-over."""
    # Output path for the combined video
    output_path = test_dir / "combined_video.mp4"

    # Combine video with voice-over
    success = combine_video_with_voiceover(
        video_path=str(test_video_path),
        voiceover_path=str(test_voiceover_path),
        output_path=str(output_path),
    )

    # Assert that the combination was successful and the file exists
    assert success, "Failed to combine video with voice-over"
    assert output_path.exists(), f"Combined video file not found at {output_path}"
