"""
Enhanced rhythmic video editor for the Women Around The World project.

This module provides advanced functionality for creating rhythmically edited videos
with beat synchronization, transitions, effects, and more.
"""

import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from moviepy.editor import VideoFileClip, concatenate_videoclips

from watw.core.video.video_utils import get_video_duration, trim_video
from watw.utils.common.logging_utils import log_execution_time, setup_logger

# Set up logger
logger = setup_logger("watw.video.enhanced_rhythmic_editor")


class EnhancedRhythmicVideoEditor:
    """
    Enhanced Rhythmic Video Editor that synchronizes video clips to music beats
    with advanced transition effects, speed controls, visual enhancements, and randomization options.
    """

    # Define transition types
    TRANSITION_TYPES = {
        "none": "No transition (hard cut)",
        "crossfade": "Smooth crossfade between clips",
        "fade": "Fade to black and back",
        "wipe_left": "Wipe from left to right",
        "wipe_right": "Wipe from right to left",
        "zoom_in": "Zoom in transition",
        "zoom_out": "Zoom out transition",
        "random": "Random selection from available transitions",  # This is handled by logic, not an FFmpeg filter
    }

    # Define visual effect types
    VISUAL_EFFECTS = {
        "none": "No effect",
        "grayscale": "Black and white effect",
        "sepia": "Sepia tone effect",
        "vibrance": "Increased color vibrance",
        "vignette": "Vignette effect (darker corners)",
        "blur_edges": "Blur edges effect",
        "sharpen": "Sharpen effect",
        "mirror": "Mirror effect (horizontal)",
        "flip": "Flip effect (vertical)",
        "random": "Random selection from available effects",  # Handled by logic
    }

    def __init__(
        self,
        input_dir: Union[str, Path],
        output_file: Union[str, Path],
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the editor with paths and configuration.

        Args:
            input_dir: Directory containing video clips
            output_file: Path for the output video file (optional)
            temp_dir: Directory for temporary files (optional)
        """
        self.start_time = time.time()  # Record start time
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)

        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="watw_rhythmic_editor_")
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Initialized RhythmicVideoEditor with temp directory: {self.temp_dir}"
        )

        # Instance variables
        self.video_clips: List[Path] = []
        self.beat_times: List[float] = []
        self.audio_duration: Optional[float] = None
        self.effective_max_time: Optional[float] = (
            None  # Actual time limit used based on audio and config["duration"]
        )
        self.clip_durations: Dict[Path, float] = {}
        self.clip_usage_count: Dict[Path, int] = defaultdict(int)
        self.clip_speeds: Dict[Path, float] = {}

    def find_video_clips(self) -> List[Path]:
        """
        Find all video clips in the input directory.

        Returns:
            List of paths to video clips
        """
        video_clips = []
        for ext in [".mp4", ".mov", ".avi", ".webm"]:
            video_clips.extend(list(self.input_dir.glob(f"*{ext}")))

        if not video_clips:
            raise FileNotFoundError(f"No video clips found in {self.input_dir}")

        logger.info(f"Found {len(video_clips)} video clips")
        return sorted(video_clips)

    def detect_beats(self) -> Tuple[List[float], float]:
        """
        Detect beats in the audio track.

        Returns:
            Tuple of (beat times in seconds, tempo in BPM)
        """
        # TODO: Implement beat detection
        # For now, return dummy values
        return [0.0, 1.0, 2.0, 3.0], 120.0

    def create_video_segments(
        self, clips: List[Path], beat_times: List[float]
    ) -> List[VideoFileClip]:
        """
        Create video segments synchronized with beats.

        Args:
            clips: List of video clip paths
            beat_times: List of beat times in seconds

        Returns:
            List of video segments
        """
        segments = []

        for i, clip_path in enumerate(clips):
            # Get video duration
            duration = get_video_duration(clip_path)

            # Trim video to match beat duration if needed
            if duration > 1.0:  # Assuming 1 second per beat for now
                trimmed_path = self.temp_dir / f"trimmed_{os.path.basename(clip_path)}"
                trim_video(
                    input_path=clip_path,
                    output_path=trimmed_path,
                    start_time=0,
                    duration=1.0,
                )
                clip_path = trimmed_path

            # Load video clip
            segment = VideoFileClip(str(clip_path))
            segments.append(segment)

        return segments

    def create_final_video(self, segments: List[VideoFileClip]) -> Path:
        """
        Create the final video by concatenating all segments.

        Args:
            segments: List of video segments

        Returns:
            Path to the final video
        """
        logger.info("Creating final video...")

        # Concatenate all segments
        final_video = concatenate_videoclips(segments)

        # Write the output file
        final_video.write_videofile(
            str(self.output_file),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
        )

        # Clean up
        final_video.close()
        for segment in segments:
            segment.close()

        logger.info(f"Final video saved to {self.output_file}")
        return self.output_file

    @log_execution_time()
    def process(self) -> Path:
        """
        Process the video editing workflow.

        Returns:
            Path to the final video
        """
        try:
            # Find video clips
            clips = self.find_video_clips()

            # Detect beats
            beat_times, tempo = self.detect_beats()

            # Create video segments
            segments = self.create_video_segments(clips, beat_times)

            # Create final video
            return self.create_final_video(segments)

        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
