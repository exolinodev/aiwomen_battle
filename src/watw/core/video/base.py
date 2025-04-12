"""
Base VideoEditor class for Women Around The World.

This module provides the base VideoEditor class that serves as the foundation
for all video editing functionality in the project.
"""

import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence

from watw.utils.common.file_utils import ensure_directory
from watw.utils.common.logging_utils import setup_logger


class VideoEditor(ABC):
    """Abstract base class for video editing operations."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the VideoEditor with optional configuration.

        Args:
            config: Configuration parameters for video editing
            temp_dir: Directory for temporary files
        """
        self.config = config or {}
        self.logger = setup_logger("watw.video_editor")
        self.temp_dir = (
            Path(temp_dir)
            if temp_dir
            else Path(tempfile.gettempdir()) / "watw_video_editor"
        )
        ensure_directory(self.temp_dir)

    @abstractmethod
    def find_video_clips(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all video clips in the specified directory.

        Args:
            extensions: List of video file extensions to search for

        Returns:
            List of paths to video clips
        """
        pass

    @abstractmethod
    def create_video(
        self, scenes: List[Dict[str, Any]], output_dir: Union[str, Path]
    ) -> Path:
        """Create a video from a list of scenes.

        Args:
            scenes: List of scene descriptions
            output_dir: Directory to save the output

        Returns:
            Path to the final video
        """
        pass

    @abstractmethod
    def combine_video_with_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        volume: Optional[float] = None,
    ) -> Path:
        """Combine a video with an audio track.

        Args:
            video_path: Path to the video file
            audio_path: Path to the audio file
            output_path: Path to save the output
            volume: Optional volume adjustment for the audio

        Returns:
            Path to the combined video
        """
        pass

    @abstractmethod
    def concatenate_videos(
        self,
        video_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        transition: Optional[str] = None,
        transition_duration: Optional[float] = None,
    ) -> Path:
        """Concatenate multiple videos into a single video.

        Args:
            video_paths: Sequence of paths to video files
            output_path: Path to save the output
            transition: Optional transition effect between videos
            transition_duration: Duration of the transition in seconds

        Returns:
            Path to the concatenated video
        """
        pass

    @abstractmethod
    def create_final_video(
        self,
        segments: List[Any],
        output_path: Union[str, Path],
        add_fade: bool = False,
        fade_duration: float = 1.0,
    ) -> Path:
        """Create the final video from segments.

        Args:
            segments: List of video segments
            output_path: Path to save the output
            add_fade: Whether to add a fade effect
            fade_duration: Duration of the fade in seconds

        Returns:
            Path to the final video
        """
        pass
