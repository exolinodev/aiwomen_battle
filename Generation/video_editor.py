import os
import subprocess
import sys

def combine_video_with_voiceover(
    video_path,
    voiceover_path,
    output_path,
    background_music_path=None,
    is_first_clip=False
):
    """
    Combine a video with a voice-over and optional background music.
    For the first clip, allows the complete voice-over to play while keeping video at 3 seconds.
    For other clips, limits both video and voice-over to 3 seconds.
    
    Args:
        video_path (str): Path to the video file
        voiceover_path (str): Path to the voice-over audio file
        output_path (str): Path to save the final video
        background_music_path (str, optional): Path to background music file
        is_first_clip (bool): Whether this is the first clip (Japan)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # First, trim the video to 3 seconds
        temp_video = output_path.replace('.mp4', '_temp.mp4')
        trim_cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-t', '3',  # Limit to 3 seconds
            '-c:v', 'copy',
            temp_video
        ]
        subprocess.run(trim_cmd, check=True)

        # For the first clip, we want to keep the complete voice-over
        if is_first_clip:
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', voiceover_path,
                '-filter_complex', '[1:a]apad=whole_dur=3[a1]',  # Pad audio to match video duration
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v',
                '-map', '[a1]',
                output_path
            ]
        else:
            # For other clips, limit both video and voice-over to 3 seconds
            cmd = [
                'ffmpeg', '-y',
                '-i', temp_video,
                '-i', voiceover_path,
                '-filter_complex', '[1:a]apad=whole_dur=3[a1]',  # Pad audio to match video duration
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v',
                '-map', '[a1]',
                '-shortest',  # Ensure output duration matches shortest input
                output_path
            ]

        subprocess.run(cmd, check=True)
        
        # Clean up temporary file
        if os.path.exists(temp_video):
            os.remove(temp_video)
            
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during video editing: {e}")
        if hasattr(e, 'output'):
            print(f"FFmpeg output: {e.output.decode()}")
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
            background_music_path=background_music_path
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

def concatenate_videos(concat_list_path, output_path, working_dir):
    """Concatenate multiple videos into a single video with transitions."""
    try:
        # Build the ffmpeg command
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-preset", "medium",
            "-crf", "23",
            output_path
        ]

        # Execute the command
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during video concatenation: {e}")
        if hasattr(e, 'output'):
            print(f"FFmpeg output: {e.output.decode()}")
        return False
    except Exception as e:
        print(f"Unexpected error during video concatenation: {e}")
        return False 