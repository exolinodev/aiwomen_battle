"""
Default video editor implementation for the Women Around The World project.

This module provides a concrete implementation of the VideoEditor base class
with standard video editing operations.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Sequence

from watw.core.video.base import VideoEditor
from watw.core.video.utils import FFmpegError, VideoFormat, validate_video_file
from watw.utils.common.logging_utils import log_execution_time, setup_logger

logger = setup_logger("watw.default_editor")


class DefaultVideoEditor(VideoEditor):
    """Default implementation of the VideoEditor base class."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the DefaultVideoEditor.

        Args:
            config: Configuration parameters for video editing
            temp_dir: Directory for temporary files
        """
        super().__init__(config, temp_dir)
        logger.info(
            f"Initialized DefaultVideoEditor with temp directory: {self.temp_dir}"
        )

    @log_execution_time()
    def find_video_clips(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all video clips in the specified directory.

        Args:
            extensions: List of video file extensions to search for

        Returns:
            List of paths to video clips
        """
        if extensions is None:
            extensions = [fmt.value for fmt in VideoFormat]

        video_clips: List[Path] = []
        for ext in extensions:
            video_clips.extend(self.temp_dir.glob(f"**/*.{ext}"))

        return [clip for clip in video_clips if validate_video_file(clip)]

    @log_execution_time()
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
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Process each scene
        processed_scenes = []
        for scene in scenes:
            # Create scene video
            scene_path = self._create_scene(scene)
            processed_scenes.append(scene_path)

        # Combine all scenes
        output_path = output_dir / "final_video.mp4"
        return self.concatenate_videos(processed_scenes, output_path)

    @log_execution_time()
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
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Use FFmpeg to combine video and audio
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-c:v",
                "copy",  # Copy video codec without re-encoding
                "-c:a",
                "aac",  # Use AAC codec for audio
                "-shortest",  # Use the duration of the shortest stream
                "-y",  # Overwrite output file if it exists
            ]

            if volume is not None:
                cmd.extend(["-filter:a", f"volume={volume}"])

            cmd.append(str(output_path))

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return output_path

        except subprocess.CalledProcessError as e:
            raise FFmpegError(
                f"Failed to combine video and audio: {e}", e.cmd, e.output
            )

    @log_execution_time()
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
        video_paths = [Path(p) for p in video_paths]
        output_path = Path(output_path)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Create a file with list of videos to concatenate
            concat_list_path = self.temp_dir / "concat_list.txt"
            with open(concat_list_path, "w") as f:
                for video_path in video_paths:
                    f.write(f"file '{video_path}'\n")

            # Use FFmpeg to concatenate videos
            cmd = [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_list_path),
                "-c",
                "copy",  # Copy streams without re-encoding
                "-y",  # Overwrite output file if it exists
            ]

            if transition:
                # Add transition effect if specified
                cmd.extend(
                    [
                        "-vf",
                        f"xfade=transition={transition}:duration={transition_duration or 1.0}",
                    ]
                )

            cmd.append(str(output_path))

            subprocess.run(cmd, capture_output=True, text=True, check=True)

            return output_path

        except subprocess.CalledProcessError as e:
            raise FFmpegError(f"Failed to concatenate videos: {e}", e.cmd, e.output)

    @log_execution_time()
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
        output_path = Path(output_path)

        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert segments to video paths
        video_paths = []
        for segment in segments:
            if isinstance(segment, (str, Path)):
                video_paths.append(Path(segment))
            elif hasattr(segment, "path"):
                video_paths.append(Path(segment.path))
            else:
                raise ValueError(f"Invalid segment type: {type(segment)}")

        # Concatenate videos with optional fade effect
        return self.concatenate_videos(
            video_paths,
            output_path,
            transition="fade" if add_fade else None,
            transition_duration=fade_duration if add_fade else None,
        )

    def _create_scene(self, scene: Dict[str, Any]) -> Path:
        """Create a single scene video.

        Args:
            scene: Scene description

        Returns:
            Path to the scene video
        """
        # Implementation details for creating a single scene
        # This is a placeholder - actual implementation would depend on scene format
        raise NotImplementedError("Scene creation not implemented")
