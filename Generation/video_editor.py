import os
import subprocess
import sys

def combine_video_with_voiceover(
    video_path,
    voiceover_path,
    output_path,
    background_music_path=None,
    music_volume=0.1
):
    """
    Combine a video with a voice-over and optional background music.
    
    Args:
        video_path (str): Path to the video file
        voiceover_path (str): Path to the voice-over audio file
        output_path (str): Path to save the final video
        background_music_path (str, optional): Path to background music file
        music_volume (float, optional): Volume of background music (0.0 to 1.0)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Base command with video and voice-over
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", voiceover_path
        ]
        
        # Add background music if provided
        if background_music_path and os.path.exists(background_music_path):
            cmd.extend(["-i", background_music_path])
            
            # Complex filter for mixing audio streams
            filter_complex = (
                f"[1:a]volume=1[a1];"  # Voice-over at full volume
                f"[2:a]volume={music_volume}[a2];"  # Background music at reduced volume
                f"[a1][a2]amix=inputs=2:duration=first[a]"  # Mix the two audio streams
            )
            
            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "0:v",  # Video from first input
                "-map", "[a]"   # Mixed audio
            ])
        else:
            # Simple audio replacement if no background music
            cmd.extend([
                "-map", "0:v",  # Video from first input
                "-map", "1:a"   # Audio from second input (voice-over)
            ])
        
        # Output options
        cmd.extend([
            "-c:v", "copy",     # Copy video codec (no re-encoding)
            "-c:a", "aac",      # Use AAC for audio
            "-shortest",        # End when shortest input ends
            output_path
        ])
        
        # Execute the command
        print(f"Executing FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("FFmpeg output:")
        print(result.stdout)
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error during video editing: {e}", file=sys.stderr)
        print(f"FFmpeg output: {e.stdout}", file=sys.stderr)
        print(f"FFmpeg error: {e.stderr}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return False

def create_final_video(
    video_clips_dir,
    voiceover_path,
    output_dir,
    background_music_path=None,
    music_volume=0.1
):
    """
    Create a final video by combining video clips with a voice-over.
    
    Args:
        video_clips_dir (str): Directory containing video clips
        voiceover_path (str): Path to the voice-over audio file
        output_dir (str): Directory to save the final video
        background_music_path (str, optional): Path to background music file
        music_volume (float, optional): Volume of background music (0.0 to 1.0)
        
    Returns:
        str: Path to the final video file, or None if failed
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all video files in the directory
        video_files = [f for f in os.listdir(video_clips_dir) 
                      if f.endswith(('.mp4', '.mov', '.avi'))]
        
        if not video_files:
            print(f"No video files found in {video_clips_dir}")
            return None
        
        # Sort video files to ensure consistent order
        video_files.sort()
        
        # Create a temporary concatenated video
        temp_concat_file = os.path.join(output_dir, "temp_concat.txt")
        with open(temp_concat_file, "w") as f:
            for video_file in video_files:
                video_path = os.path.join(video_clips_dir, video_file)
                f.write(f"file '{video_path}'\n")
        
        # Concatenate videos
        temp_video_path = os.path.join(output_dir, "temp_concatenated.mp4")
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_concat_file,
            "-c", "copy",
            temp_video_path
        ]
        
        print(f"Concatenating videos: {' '.join(concat_cmd)}")
        subprocess.run(concat_cmd, check=True)
        
        # Combine with voice-over
        final_video_path = os.path.join(output_dir, "final_video.mp4")
        success = combine_video_with_voiceover(
            video_path=temp_video_path,
            voiceover_path=voiceover_path,
            output_path=final_video_path,
            background_music_path=background_music_path,
            music_volume=music_volume
        )
        
        # Clean up temporary files
        if os.path.exists(temp_concat_file):
            os.remove(temp_concat_file)
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        if success:
            return final_video_path
        else:
            return None
    
    except Exception as e:
        print(f"Error creating final video: {e}", file=sys.stderr)
        return None 