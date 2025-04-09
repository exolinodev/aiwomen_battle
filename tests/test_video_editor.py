#!/usr/bin/env python3
import os
import sys
import shutil
from pathlib import Path

def test_video_editor():
    """Test the video editor to combine a video with a voice-over."""
    try:
        # Import the video editor module
        sys.path.append(os.path.dirname(__file__))
        from Generation.video_editor import combine_video_with_voiceover
        
        # Create test directories
        test_dir = "test_video_editor"
        os.makedirs(test_dir, exist_ok=True)
        
        # Check if we have a test video and voice-over
        test_video_path = os.path.join("test_output", "test_video.mp4")
        test_voiceover_path = os.path.join("test_output", "voiceover_1.mp3")
        
        # If test video doesn't exist, create a dummy video
        if not os.path.exists(test_video_path):
            print("Test video not found. Creating a dummy video...")
            # Create a simple video using FFmpeg
            os.system(f"ffmpeg -y -f lavfi -i color=c=blue:s=1280x720:d=5 -c:v libx264 {test_video_path}")
        
        # Check if voice-over exists
        if not os.path.exists(test_voiceover_path):
            print("Voice-over not found. Please run test_voiceover_only.py first.")
            assert False, "Voice-over file not found"
        
        # Output path for the combined video
        output_path = os.path.join(test_dir, "combined_video.mp4")
        
        # Combine video with voice-over
        print(f"Combining video with voice-over...")
        success = combine_video_with_voiceover(
            video_path=test_video_path,
            voiceover_path=test_voiceover_path,
            output_path=output_path
        )
        
        # Assert that the combination was successful and the file exists
        assert success, "Failed to combine video with voice-over"
        assert os.path.exists(output_path), f"Combined video file not found at {output_path}"
        print(f"Successfully combined video with voice-over: {output_path}")
    
    except Exception as e:
        print(f"Error testing video editor: {e}")
        raise  # Re-raise the exception to fail the test

if __name__ == "__main__":
    print("Testing video editor...")
    try:
        test_video_editor()
        print("\nVideo editor test passed!")
    except:
        print("\nVideo editor test failed!") 