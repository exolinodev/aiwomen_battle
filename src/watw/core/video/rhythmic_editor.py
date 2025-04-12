"""
Rhythmic video editor for the Women Around The World project.

This module provides functionality to create rhythmically edited videos
by synchronizing video cuts with audio beats.
"""

import argparse
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import librosa
import numpy as np
from scipy.signal import find_peaks

from watw.core.video.utils import (
    FFmpegError,
    VideoFile,
    VideoOperationError,
    build_ffmpeg_command,
    create_temp_file,
    get_video_duration,
    run_ffmpeg_command,
)
from watw.utils.common.logging_utils import setup_logger

# Configure logging
logger = setup_logger("watw.rhythmic_editor")


@dataclass
class VideoSegment:
    """Represents a segment of a video."""

    start_time: float
    duration: float
    video_path: Path
    transition: Optional[str] = None


class RhythmicVideoEditor:
    """Class for creating rhythmically edited videos."""

    def __init__(
        self,
        clips_dir: Union[str, Path],
        audio_path: Union[str, Path],
        output_file: Optional[Union[str, Path]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the RhythmicVideoEditor.

        Args:
            clips_dir: Directory containing video clips
            audio_path: Path to the audio file for beat detection
            output_file: Path to save the final video (optional)
            temp_dir: Directory for temporary files (optional)
        """
        self.clips_dir = Path(clips_dir)
        self.audio_path = Path(audio_path)

        if output_file is None:
            output_file = self.clips_dir.parent / "final_rhythmic_video.mp4"
        self.output_file = Path(output_file)

        if temp_dir is None:
            temp_dir = self.clips_dir.parent / "temp"
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize lists for video processing
        self.video_clips: List[VideoFile] = []
        self.beat_times: List[float] = []
        self.segments: List[VideoSegment] = []

        logger.info(f"Initialized RhythmicVideoEditor with clips_dir: {self.clips_dir}")
        logger.info(f"Audio path: {self.audio_path}")
        logger.info(f"Output file: {self.output_file}")
        logger.info(f"Temp directory: {self.temp_dir}")

    def find_video_clips(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find video clips in the clips directory.

        Args:
            extensions: List of video file extensions to look for

        Returns:
            List of paths to video clips
        """
        if extensions is None:
            extensions = [".mp4", ".mov", ".avi", ".webm"]

        clips: List[Path] = []
        for ext in extensions:
            clips.extend(self.clips_dir.glob(f"*{ext}"))
        return sorted(clips)

    def detect_beats(self, sensitivity: float = 0.5) -> List[float]:
        """Detect beats in the audio file.

        Args:
            sensitivity: Beat detection sensitivity (0.0 to 1.0)

        Returns:
            List of beat times in seconds
        """
        try:
            # Load audio file
            y, sr = librosa.load(str(self.audio_path))

            # Get tempo and beat frames
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

            # Convert beat frames to times
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # Adjust sensitivity
            if sensitivity != 1.0:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                peaks, _ = find_peaks(onset_env, height=sensitivity)
                beat_times = librosa.frames_to_time(peaks, sr=sr)

            return cast(List[float], beat_times.tolist())
        except Exception as e:
            raise VideoOperationError(f"Failed to detect beats: {e}")

    def create_segments(
        self, min_segment_duration: float = 0.5, max_segment_duration: float = 2.0
    ) -> List[VideoSegment]:
        """Create video segments based on beat times.

        Args:
            min_segment_duration: Minimum duration for a segment
            max_segment_duration: Maximum duration for a segment

        Returns:
            List of video segments
        """
        if not self.video_clips:
            self.video_clips = [VideoFile(p) for p in self.find_video_clips()]

        if not self.beat_times:
            self.beat_times = self.detect_beats()

        segments = []
        current_time = 0.0
        clip_index = 0

        for i in range(len(self.beat_times) - 1):
            start_time = self.beat_times[i]
            end_time = self.beat_times[i + 1]
            duration = end_time - start_time

            if duration < min_segment_duration:
                continue
            if duration > max_segment_duration:
                # Split long segments
                num_splits = int(duration / max_segment_duration) + 1
                split_duration = duration / num_splits
                for j in range(num_splits):
                    segment = VideoSegment(
                        start_time=start_time + j * split_duration,
                        duration=split_duration,
                        video_path=self.video_clips[clip_index].path,
                    )
                    segments.append(segment)
                    clip_index = (clip_index + 1) % len(self.video_clips)
            else:
                segment = VideoSegment(
                    start_time=start_time,
                    duration=duration,
                    video_path=self.video_clips[clip_index].path,
                )
                segments.append(segment)
                clip_index = (clip_index + 1) % len(self.video_clips)

        self.segments = segments
        logger.info(f"Created {len(self.segments)} segments")

        return self.segments

    def create_final_video(self) -> Path:
        """Create the final rhythmically edited video.

        Returns:
            Path to the final video file
        """
        if not self.segments:
            self.segments = self.create_segments()

        # Create temporary files for each segment
        temp_files = []
        for segment in self.segments:
            temp_file = create_temp_file(suffix=".mp4")
            try:
                cmd = build_ffmpeg_command(
                    input_paths=segment.video_path,
                    output_path=temp_file,
                    filters=[
                        f"trim=start={segment.start_time}:duration={segment.duration}"
                    ],
                )
                run_ffmpeg_command(cmd)
                temp_files.append(temp_file)
            except FFmpegError as e:
                logger.error(f"Failed to process segment: {e}")
                continue

        # Concatenate all segments
        try:
            cmd = build_ffmpeg_command(
                input_paths=temp_files,
                output_path=self.output_file,
                filters=["concat=n={}:v=1:a=0".format(len(temp_files))],
            )
            run_ffmpeg_command(cmd)
        except FFmpegError as e:
            raise VideoOperationError(f"Failed to create final video: {e}")
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass

        logger.info(f"Final video created: {self.output_file}")

        return self.output_file


def main() -> None:
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create a rhythmically edited video")
    parser.add_argument("clips_dir", help="Directory containing video clips")
    parser.add_argument("audio_path", help="Path to the audio file for beat detection")
    parser.add_argument("--output", "-o", help="Path to save the final video")
    parser.add_argument("--temp-dir", "-t", help="Directory for temporary files")
    parser.add_argument(
        "--sensitivity",
        "-s",
        type=float,
        default=0.5,
        help="Sensitivity of beat detection (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-duration",
        "-m",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds",
    )
    parser.add_argument(
        "--max-duration",
        "-M",
        type=float,
        default=2.0,
        help="Maximum segment duration in seconds",
    )

    args = parser.parse_args()

    try:
        editor = RhythmicVideoEditor(
            clips_dir=args.clips_dir,
            audio_path=args.audio_path,
            output_file=args.output,
            temp_dir=args.temp_dir,
        )

        editor.find_video_clips()
        editor.detect_beats(sensitivity=args.sensitivity)
        editor.create_segments(
            min_segment_duration=args.min_duration,
            max_segment_duration=args.max_duration,
        )
        editor.create_final_video()

        print(f"Final video created: {editor.output_file}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
