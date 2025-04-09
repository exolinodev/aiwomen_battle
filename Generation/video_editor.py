"""
Video editing module for the Women Around The World project.

This module provides a structured way to perform video operations such as
combining videos with audio, concatenating multiple videos, and creating
final videos with voiceovers and background music.
"""

import os
import subprocess
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class VideoOperationError(Exception):
    """Base exception for video operation errors."""
    pass

class FFmpegError(VideoOperationError):
    """Exception raised when FFmpeg operations fail."""
    def __init__(self, message: str, command: List[str], output: Optional[str] = None):
        self.message = message
        self.command = command
        self.output = output
        super().__init__(f"{message} (Command: {' '.join(command)})")

class VideoValidationError(VideoOperationError):
    """Exception raised when video validation fails."""
    pass

class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"

@dataclass
class VideoMetadata:
    """Metadata for a video file."""
    duration: float
    width: int
    height: int
    format: str
    codec: str
    bitrate: int
    fps: float
    size: int

class VideoFile:
    """
    Represents a video file with operations and metadata.
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize a VideoFile instance.
        
        Args:
            path: Path to the video file
        """
        self.path = Path(path)
        self._metadata: Optional[VideoMetadata] = None
        
    def exists(self) -> bool:
        """Check if the video file exists."""
        return self.path.exists()
        
    def get_metadata(self) -> VideoMetadata:
        """
        Get metadata for the video file.
        
        Returns:
            VideoMetadata: Video metadata
            
        Raises:
            VideoOperationError: If metadata cannot be retrieved
        """
        if self._metadata is not None:
            return self._metadata
            
        if not self.exists():
            raise VideoOperationError(f"Video file does not exist: {self.path}")
            
        try:
            # Use FFprobe to get video metadata
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,codec_name,avg_frame_rate,bit_rate',
                '-show_entries', 'format=duration,size,format_name',
                '-of', 'json',
                str(self.path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = eval(result.stdout)
            
            # Extract metadata from FFprobe output
            stream = data['streams'][0]
            format_info = data['format']
            
            # Parse frame rate (format: "num/den")
            fps_str = stream['avg_frame_rate']
            fps_num, fps_den = map(int, fps_str.split('/'))
            fps = fps_num / fps_den if fps_den != 0 else 0
            
            self._metadata = VideoMetadata(
                duration=float(format_info['duration']),
                width=int(stream['width']),
                height=int(stream['height']),
                format=format_info['format_name'],
                codec=stream['codec_name'],
                bitrate=int(stream.get('bit_rate', 0)),
                fps=fps,
                size=int(format_info['size'])
            )
            
            return self._metadata
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to get video metadata", cmd, e.output)
        except Exception as e:
            raise VideoOperationError(f"Error getting video metadata: {str(e)}")
    
    def validate(self, min_duration: float = 0.0, max_duration: float = 3600.0,
                min_width: int = 0, min_height: int = 0) -> bool:
        """
        Validate the video file.
        
        Args:
            min_duration: Minimum duration in seconds
            max_duration: Maximum duration in seconds
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            VideoValidationError: If validation fails
        """
        try:
            metadata = self.get_metadata()
            
            if metadata.duration < min_duration or metadata.duration > max_duration:
                raise VideoValidationError(
                    f"Video duration {metadata.duration}s is outside allowed range "
                    f"[{min_duration}s, {max_duration}s]"
                )
                
            if metadata.width < min_width or metadata.height < min_height:
                raise VideoValidationError(
                    f"Video dimensions {metadata.width}x{metadata.height} are below "
                    f"minimum required {min_width}x{min_height}"
                )
                
            return True
            
        except VideoOperationError as e:
            raise VideoValidationError(f"Video validation failed: {str(e)}")
    
    def trim(self, start_time: float, duration: float, output_path: Union[str, Path]) -> 'VideoFile':
        """
        Trim the video to a specific duration.
        
        Args:
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Path to save the trimmed video
            
        Returns:
            VideoFile: The trimmed video file
            
        Raises:
            FFmpegError: If trimming fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'copy',
                '-c:a', 'copy',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return VideoFile(output_path)
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to trim video", cmd, e.output.decode())
    
    def extract_audio(self, output_path: Union[str, Path]) -> Path:
        """
        Extract audio from the video.
        
        Args:
            output_path: Path to save the audio
            
        Returns:
            Path: Path to the extracted audio
            
        Raises:
            FFmpegError: If extraction fails
        """
        output_path = Path(output_path)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.path),
                '-vn',  # No video
                '-acodec', 'libmp3lame',
                '-q:a', '2',  # High quality
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to extract audio", cmd, e.output.decode())

class AudioFile:
    """
    Represents an audio file with operations.
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize an AudioFile instance.
        
        Args:
            path: Path to the audio file
        """
        self.path = Path(path)
        
    def exists(self) -> bool:
        """Check if the audio file exists."""
        return self.path.exists()
    
    def get_duration(self) -> float:
        """
        Get the duration of the audio file.
        
        Returns:
            float: Duration in seconds
            
        Raises:
            VideoOperationError: If duration cannot be retrieved
        """
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                str(self.path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return float(result.stdout.strip())
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to get audio duration", cmd, e.output)
        except Exception as e:
            raise VideoOperationError(f"Error getting audio duration: {str(e)}")
    
    def trim(self, start_time: float, duration: float, output_path: Union[str, Path]) -> Path:
        """
        Trim the audio to a specific duration.
        
        Args:
            start_time: Start time in seconds
            duration: Duration in seconds
            output_path: Path to save the trimmed audio
            
        Returns:
            Path: Path to the trimmed audio
            
        Raises:
            FFmpegError: If trimming fails
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.path),
                '-ss', str(start_time),
                '-t', str(duration),
                '-acodec', 'libmp3lame',
                '-q:a', '2',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to trim audio", cmd, e.output.decode())
    
    def adjust_volume(self, volume_factor: float, output_path: Union[str, Path]) -> Path:
        """
        Adjust the volume of the audio.
        
        Args:
            volume_factor: Volume adjustment factor (0.0 to 1.0)
            output_path: Path to save the adjusted audio
            
        Returns:
            Path: Path to the adjusted audio
            
        Raises:
            FFmpegError: If adjustment fails
        """
        output_path = Path(output_path)
        
        try:
            cmd = [
                'ffmpeg', '-y',
                '-i', str(self.path),
                '-filter:a', f'volume={volume_factor}',
                '-acodec', 'libmp3lame',
                '-q:a', '2',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to adjust audio volume", cmd, e.output.decode())

class VideoEditor:
    """
    Provides high-level video editing operations.
    """
    
    def __init__(self, temp_dir: Optional[Union[str, Path]] = None):
        """
        Initialize a VideoEditor instance.
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "watw_video_editor"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    def combine_video_with_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        video_duration: Optional[float] = None,
        audio_pad: bool = False,
        audio_pad_duration: Optional[float] = None
    ) -> VideoFile:
        """
        Combine a video with audio.
        
        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path to save the combined video
            video_duration: Optional duration to trim the video to
            audio_pad: Whether to pad the audio to match the video duration
            audio_pad_duration: Optional duration to pad the audio to
            
        Returns:
            VideoFile: The combined video file
            
        Raises:
            VideoOperationError: If combination fails
        """
        video = VideoFile(video_path)
        audio = AudioFile(audio_path)
        output_path = Path(output_path)
        
        # Validate inputs
        if not video.exists():
            raise VideoOperationError(f"Video file does not exist: {video_path}")
        if not audio.exists():
            raise VideoOperationError(f"Audio file does not exist: {audio_path}")
            
        # Create temporary files
        temp_video = self.temp_dir / f"temp_video_{output_path.stem}.mp4"
        temp_audio = self.temp_dir / f"temp_audio_{output_path.stem}.mp3"
        
        try:
            # Trim video if needed
            if video_duration is not None:
                video = video.trim(0, video_duration, temp_video)
            else:
                temp_video = video.path
                
            # Prepare audio
            if audio_pad:
                pad_duration = audio_pad_duration or video.get_metadata().duration
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(audio.path),
                    '-filter:a', f'apad=whole_dur={pad_duration}',
                    '-acodec', 'libmp3lame',
                    '-q:a', '2',
                    str(temp_audio)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
            else:
                temp_audio = audio.path
                
            # Combine video and audio
            cmd = [
                'ffmpeg', '-y',
                '-i', str(temp_video),
                '-i', str(temp_audio),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v',
                '-map', '1:a',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return VideoFile(output_path)
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to combine video with audio", cmd, e.output.decode())
        except Exception as e:
            raise VideoOperationError(f"Error combining video with audio: {str(e)}")
        finally:
            # Clean up temporary files
            if temp_video != video.path and temp_video.exists():
                temp_video.unlink()
            if temp_audio != audio.path and temp_audio.exists():
                temp_audio.unlink()
    
    def combine_video_with_voiceover(
        self,
        video_path: Union[str, Path],
        voiceover_path: Union[str, Path],
        output_path: Union[str, Path],
        background_music_path: Optional[Union[str, Path]] = None,
        is_first_clip: bool = False,
        clip_duration: float = 3.0,
        music_volume: float = 0.1
    ) -> VideoFile:
        """
        Combine a video with a voice-over and optional background music.
        
        Args:
            video_path: Path to the video file
            voiceover_path: Path to the voice-over audio file
            output_path: Path to save the combined video
            background_music_path: Optional path to background music file
            is_first_clip: Whether this is the first clip
            clip_duration: Duration to trim the video to
            music_volume: Volume of background music (0.0 to 1.0)
            
        Returns:
            VideoFile: The combined video file
            
        Raises:
            VideoOperationError: If combination fails
        """
        video = VideoFile(video_path)
        voiceover = AudioFile(voiceover_path)
        output_path = Path(output_path)
        
        # Validate inputs
        if not video.exists():
            raise VideoOperationError(f"Video file does not exist: {video_path}")
        if not voiceover.exists():
            raise VideoOperationError(f"Voice-over file does not exist: {voiceover_path}")
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Combine video with voice-over
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video.path),
                '-i', str(voiceover.path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-map', '0:v',
                '-map', '1:a',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return VideoFile(output_path)
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to combine video with voice-over", cmd, e.output.decode())
        except Exception as e:
            raise VideoOperationError(f"Error combining video with voice-over: {str(e)}")
    
    def concatenate_videos(
        self,
        video_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        validate_dimensions: bool = True
    ) -> VideoFile:
        """
        Concatenate multiple videos.
        
        Args:
            video_paths: List of paths to video files
            output_path: Path to save the concatenated video
            validate_dimensions: Whether to validate that all videos have the same dimensions
            
        Returns:
            VideoFile: The concatenated video file
            
        Raises:
            VideoOperationError: If concatenation fails
        """
        if not video_paths:
            raise VideoOperationError("No videos provided for concatenation")
            
        output_path = Path(output_path)
        
        # Validate all videos exist
        for path in video_paths:
            if not Path(path).exists():
                raise VideoOperationError(f"Video file does not exist: {path}")
        
        # Create a file listing all videos to concatenate
        concat_list_path = self.temp_dir / f"concat_list_{output_path.stem}.txt"
        
        try:
            # Check dimensions if required
            if validate_dimensions:
                first_video = VideoFile(video_paths[0])
                first_metadata = first_video.get_metadata()
                
                for path in video_paths[1:]:
                    video = VideoFile(path)
                    metadata = video.get_metadata()
                    
                    if (metadata.width != first_metadata.width or 
                        metadata.height != first_metadata.height):
                        raise VideoValidationError(
                            f"Video dimensions mismatch: {path} has {metadata.width}x{metadata.height}, "
                            f"expected {first_metadata.width}x{first_metadata.height}"
                        )
            
            # Create concat list file
            with open(concat_list_path, 'w') as f:
                for path in video_paths:
                    f.write(f"file '{path}'\n")
            
            # Concatenate videos
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list_path),
                '-c', 'copy',
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return VideoFile(output_path)
            
        except subprocess.CalledProcessError as e:
            raise FFmpegError("Failed to concatenate videos", cmd, e.output.decode())
        except Exception as e:
            raise VideoOperationError(f"Error concatenating videos: {str(e)}")
        finally:
            # Clean up temporary files
            if concat_list_path.exists():
                concat_list_path.unlink()
    
    def create_final_video(
        self,
        video_clips_dir: Union[str, Path],
        voiceover_path: Union[str, Path],
        output_dir: Union[str, Path],
        background_music_path: Optional[Union[str, Path]] = None,
        music_volume: float = 0.1,
        clip_duration: float = 3.0
    ) -> VideoFile:
        """
        Create a final video by combining multiple video clips with a voice-over.
        
        Args:
            video_clips_dir: Directory containing video clips
            voiceover_path: Path to the voice-over audio file
            output_dir: Directory to save the final video
            background_music_path: Optional path to background music file
            music_volume: Volume of background music (0.0 to 1.0)
            clip_duration: Duration for each clip
            
        Returns:
            VideoFile: The final video file
            
        Raises:
            VideoOperationError: If creation fails
        """
        video_clips_dir = Path(video_clips_dir)
        output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of video clips
        video_clips = []
        for file in video_clips_dir.glob('*.mp4'):
            video_clips.append(file)
        
        if not video_clips:
            raise VideoOperationError(f"No video clips found in {video_clips_dir}")
        
        # Sort video clips by name
        video_clips.sort()
        
        # Create a temporary directory for processed clips
        temp_dir = output_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Process each video clip
        processed_clips = []
        for i, clip in enumerate(video_clips):
            # Determine if this is the first clip
            is_first_clip = (i == 0)
            
            # Create output path for this clip
            output_path = temp_dir / f"processed_{clip.stem}.mp4"
            
            try:
                # Combine video with voice-over
                processed_video = self.combine_video_with_voiceover(
                    video_path=clip,
                    voiceover_path=voiceover_path,
                    output_path=output_path,
                    background_music_path=background_music_path,
                    is_first_clip=is_first_clip,
                    clip_duration=clip_duration,
                    music_volume=music_volume
                )
                
                processed_clips.append(processed_video.path)
                logger.info(f"Processed clip {i+1}/{len(video_clips)}: {clip.name}")
                
            except VideoOperationError as e:
                logger.error(f"Failed to process clip {clip.name}: {str(e)}")
                continue
        
        if not processed_clips:
            raise VideoOperationError("No clips were successfully processed")
        
        # Final output path
        final_output_path = output_dir / "final_video.mp4"
        
        # Concatenate all processed clips
        try:
            final_video = self.concatenate_videos(processed_clips, final_output_path)
            logger.info(f"Final video created: {final_output_path}")
            
            return final_video
            
        except VideoOperationError as e:
            raise VideoOperationError(f"Failed to create final video: {str(e)}")
        finally:
            # Clean up temporary directory
            if temp_dir.exists():
                for file in temp_dir.glob('*'):
                    file.unlink()
                temp_dir.rmdir()

# For backward compatibility
def combine_video_with_voiceover(
    video_path: Union[str, Path],
    voiceover_path: Union[str, Path],
    output_path: Union[str, Path],
    background_music_path: Optional[Union[str, Path]] = None,
    is_first_clip: bool = False
) -> bool:
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
        # Convert paths to Path objects
        video_path = Path(video_path)
        voiceover_path = Path(voiceover_path)
        output_path = Path(output_path)
        
        # Validate input files exist
        if not video_path.exists():
            logger.error(f"Video file does not exist: {video_path}")
            return False
        if not voiceover_path.exists():
            logger.error(f"Voice-over file does not exist: {voiceover_path}")
            return False
            
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create VideoEditor instance
        editor = VideoEditor()
        
        # Combine video with voice-over
        editor.combine_video_with_voiceover(
            video_path=str(video_path),
            voiceover_path=str(voiceover_path),
            output_path=str(output_path),
            background_music_path=str(background_music_path) if background_music_path else None,
            is_first_clip=is_first_clip
        )
        
        # Verify output file was created
        if not output_path.exists():
            logger.error(f"Failed to create output file: {output_path}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error combining video with voice-over: {str(e)}")
        return False

def create_final_video(
    video_clips_dir: Union[str, Path],
    voiceover_path: Union[str, Path],
    output_dir: Union[str, Path],
    background_music_path: Optional[Union[str, Path]] = None,
    music_volume: float = 0.1
) -> Optional[str]:
    """
    Create a final video by combining multiple video clips with a voice-over.
    
    Args:
        video_clips_dir (str): Directory containing video clips
        voiceover_path (str): Path to the voice-over audio file
        output_dir (str): Directory to save the final video
        background_music_path (str, optional): Path to background music file
        music_volume (float): Volume of background music (0.0 to 1.0)
        
    Returns:
        str: Path to the final video, or None if failed
    """
    try:
        editor = VideoEditor()
        final_video = editor.create_final_video(
            video_clips_dir=video_clips_dir,
            voiceover_path=voiceover_path,
            output_dir=output_dir,
            background_music_path=background_music_path,
            music_volume=music_volume
        )
        return str(final_video.path)
    except Exception as e:
        logger.error(f"Error creating final video: {str(e)}")
        return None

def concatenate_videos(
    concat_list_path: Union[str, Path],
    output_path: Union[str, Path],
    working_dir: Union[str, Path]
) -> bool:
    """
    Concatenate multiple videos using FFmpeg.
    
    Args:
        concat_list_path (str): Path to the file listing videos to concatenate
        output_path (str): Path to save the concatenated video
        working_dir (str): Working directory for temporary files
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        editor = VideoEditor(temp_dir=working_dir)
        
        # Read the concat list file
        with open(concat_list_path, 'r') as f:
            video_paths = [line.strip().replace("file '", "").replace("'", "") for line in f if line.strip()]
        
        editor.concatenate_videos(video_paths, output_path)
        return True
    except Exception as e:
        logger.error(f"Error concatenating videos: {str(e)}")
        return False 