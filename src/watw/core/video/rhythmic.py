"""
RhythmicVideoEditor for Women Around The World.

This module provides the RhythmicVideoEditor class that extends the base VideoEditor
with capabilities for creating rhythmically synchronized videos.
"""

import glob
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast, Sequence

import librosa
import numpy as np
from scipy.signal import find_peaks

from watw.core.video.base import VideoEditor
from watw.utils.video.ffmpeg import (
    build_ffmpeg_command,
    run_ffmpeg_command,
    build_trim_command,
)
from watw.utils.video.metadata import get_video_duration


@dataclass
class BeatInfo:
    """Information about detected beats in an audio file."""

    tempo: float
    beat_times: List[float]
    duration: float


class RhythmicVideoEditor(VideoEditor):
    """Video editor for creating rhythmically synchronized videos."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the RhythmicVideoEditor.

        Args:
            config: Configuration parameters
            temp_dir: Directory for temporary files
        """
        super().__init__(config, temp_dir)
        config_dict = config or {}
        self.clips_dir = Path(config_dict.get("clips_dir", "."))
        self.audio_path = Path(config_dict.get("audio_path", ""))
        self.video_clips: List[Path] = []
        self.beat_info: Optional[BeatInfo] = None
        self.clip_usage_count: Dict[Path, int] = {}

        # Default configuration
        self.config.setdefault("min_clip_duration", 0.5)
        self.config.setdefault("max_clip_duration", 5.0)
        self.config.setdefault("transition_duration", 0.5)
        self.config.setdefault("avoid_clip_repetition", True)
        self.config.setdefault("max_clip_repeats", 3)
        self.config.setdefault("quality", "high")
        self.config.setdefault("video_codec", "libx264")
        self.config.setdefault("pix_fmt", "yuv420p")

        # Quality presets
        self.QUALITY_PRESETS = {
            "low": {"preset": "ultrafast", "crf": 28, "video_bitrate": "2M"},
            "medium": {"preset": "fast", "crf": 23, "video_bitrate": "4M"},
            "high": {"preset": "medium", "crf": 18, "video_bitrate": "8M"},
            "ultra": {"preset": "slow", "crf": 15, "video_bitrate": "16M"},
        }

    def find_video_clips(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all video clips in the specified directory.

        Args:
            extensions: List of video file extensions to search for

        Returns:
            List of paths to video clips
        """
        if extensions is None:
            extensions = [".mp4", ".mov", ".avi", ".mkv", ".wmv"]

        self.video_clips = []
        self.logger.info(
            f"Searching for video clips with extensions {extensions} in '{self.clips_dir}'..."
        )

        for ext in extensions:
            # Case-insensitive search using glob patterns for both lower and upper case
            pattern_lower = os.path.join(self.clips_dir, f"*{ext.lower()}")
            pattern_upper = os.path.join(self.clips_dir, f"*{ext.upper()}")
            found_lower = [Path(f) for f in glob.glob(pattern_lower)]
            found_upper = [Path(f) for f in glob.glob(pattern_upper)]
            self.video_clips.extend(found_lower)
            # Only add upper case results if they are different files (e.g. case-sensitive filesystem)
            self.video_clips.extend([f for f in found_upper if f not in found_lower])

        # Remove duplicates and sort
        self.video_clips = sorted(list(set(self.video_clips)))

        if not self.video_clips:
            # Be more specific about why no clips were found
            if not os.path.isdir(self.clips_dir):
                raise FileNotFoundError(
                    f"Input directory does not exist: {self.clips_dir}"
                )
            else:
                raise FileNotFoundError(
                    f"No video clips found in directory '{self.clips_dir}' with extensions {extensions}"
                )

        self.logger.info(f"Found {len(self.video_clips)} potential video clips.")
        if len(self.video_clips) > 20:
            self.logger.info("Listing first 10:")
            for i, clip in enumerate(self.video_clips[:10]):
                self.logger.info(f" - {os.path.basename(clip)}")
            self.logger.info(f" - ... and {len(self.video_clips) - 10} more.")
        else:
            for clip in self.video_clips:
                self.logger.info(f" - {os.path.basename(clip)}")

        return self.video_clips

    def detect_beats(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        sensitivity: float = 1.2,
        min_beats: int = 10,
        min_bpm: float = 60.0,
        max_bpm: float = 180.0,
    ) -> BeatInfo:
        """Detect beats in an audio file using multiple methods.

        Args:
            audio_path: Path to the audio file (uses self.audio_path if None)
            sensitivity: Sensitivity multiplier for beat detection
            min_beats: Minimum number of beats to detect
            min_bpm: Minimum tempo in beats per minute
            max_bpm: Maximum tempo in beats per minute

        Returns:
            BeatInfo object containing beat information
        """
        if audio_path is None:
            audio_path = self.audio_path
        else:
            audio_path = Path(audio_path)

        if not audio_path or not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self.logger.info(f"Analyzing audio file: {audio_path.name}")

        try:
            # Load audio with increased precision
            y, sr = librosa.load(str(audio_path), sr=44100)

            # Combine multiple beat detection methods for better results
            strengths = []

            # 1. Onset detection
            self.logger.info("Method 1: Onset detection...")
            onset_env = librosa.onset.onset_strength(
                y=y, sr=sr, hop_length=512, aggregate=np.median
            )

            # Dynamic threshold based on percentiles
            threshold = np.percentile(onset_env, 75) * sensitivity
            peaks, peak_strengths = find_peaks(
                onset_env,
                height=threshold,
                distance=sr
                / 512
                / 4,  # Minimum distance between peaks: 1/4 beat at 120 BPM
            )
            beat_times_1 = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
            strengths.extend(peak_strengths["peak_heights"])

            # 2. Tempogram-based beat detection
            self.logger.info("Method 2: Tempogram analysis...")
            tempo, beats_frames = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=512, tightness=100, start_bpm=120, trim=True
            )
            beat_times_2 = librosa.frames_to_time(beats_frames, sr=sr, hop_length=512)
            strengths.extend(
                [1.0] * len(beat_times_2)
            )  # Default strength for tempogram beats

            # 3. Spectral flux-based beat detection
            self.logger.info("Method 3: Spectral flux analysis...")
            spec = np.abs(librosa.stft(y, hop_length=512))
            spec_flux = np.sum(np.diff(spec, axis=0), axis=0)
            spec_flux = np.maximum(spec_flux, 0.0)

            # Normalize and apply moving average
            spec_flux = spec_flux / np.max(spec_flux)
            window_size = 5
            weights = np.hamming(window_size)
            spec_flux_smooth = np.convolve(
                spec_flux, weights / weights.sum(), mode="same"
            )

            # Threshold based on smoothed spectral flux
            sf_threshold = np.percentile(spec_flux_smooth, 75) * sensitivity
            sf_peaks, sf_strengths = find_peaks(
                spec_flux_smooth, height=sf_threshold, distance=sr / 512 / 4
            )
            beat_times_3 = librosa.frames_to_time(sf_peaks, sr=sr, hop_length=512)
            strengths.extend(sf_strengths["peak_heights"])

            # 4. RMS energy-based beat detection (good for bass beats)
            self.logger.info("Method 4: Energy-based detection...")
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            rms_threshold = np.percentile(rms, 75) * sensitivity
            rms_peaks, rms_strengths = find_peaks(
                rms, height=rms_threshold, distance=sr / 512 / 4
            )
            beat_times_4 = librosa.frames_to_time(rms_peaks, sr=sr, hop_length=512)
            strengths.extend(rms_strengths["peak_heights"])

            # Intelligent combination of all methods
            self.logger.info("Combining results from all methods...")

            # Collect all beats in a list
            all_beats = np.concatenate(
                [beat_times_1, beat_times_2, beat_times_3, beat_times_4]
            )
            all_strengths = np.array(strengths)

            # Group nearby beats (clustering)
            all_beats = np.sort(all_beats)
            grouped_beats = []
            grouped_strengths = []
            last_beat = -1
            min_beat_distance = 0.1  # 100ms minimum distance

            for i, beat in enumerate(all_beats):
                if last_beat == -1 or beat - last_beat >= min_beat_distance:
                    grouped_beats.append(beat)
                    grouped_strengths.append(all_strengths[i])
                    last_beat = beat

            beat_times = np.array(grouped_beats)
            beat_strengths = np.array(grouped_strengths)

            # Determine dominant tempo and fill gaps
            if len(beat_times) >= 2:
                # Calculate intervals between beats
                intervals = np.diff(beat_times)
                tempo = 60.0 / np.median(intervals)

                # Ensure tempo is within bounds
                if tempo < min_bpm:
                    tempo = min_bpm
                elif tempo > max_bpm:
                    tempo = max_bpm

                # Fill gaps in beat sequence
                filled_beats = []
                for i in range(len(beat_times) - 1):
                    filled_beats.append(beat_times[i])
                    interval = beat_times[i + 1] - beat_times[i]
                    if interval > 1.5 * (60.0 / tempo):
                        # Add intermediate beats
                        num_intermediate = int(interval / (60.0 / tempo)) - 1
                        for j in range(1, num_intermediate + 1):
                            filled_beats.append(
                                beat_times[i] + j * (interval / (num_intermediate + 1))
                            )
                filled_beats.append(beat_times[-1])
                beat_times = np.array(filled_beats)

            else:
                tempo = 120.0  # Default tempo if not enough beats

            # Ensure minimum number of beats
            if len(beat_times) < min_beats:
                raise ValueError(
                    f"Not enough beats detected (found {len(beat_times)}, minimum {min_beats})"
                )

            # Store beat information
            self.beat_info = BeatInfo(
                tempo=float(tempo),
                beat_times=beat_times.tolist(),
                duration=float(len(y) / sr),
            )

            self.logger.info(
                f"Detected {len(beat_times)} beats at {tempo:.1f} BPM over {len(y) / sr:.1f} seconds"
            )

            return self.beat_info

        except Exception as e:
            self.logger.error(f"Error detecting beats: {str(e)}")
            raise

    def _select_clip(
        self,
        previous_clip: Optional[Union[str, Path]] = None,
        sequence_position: int = 0,
        total_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Select a clip for a segment.

        Args:
            previous_clip: Path to the previous clip
            sequence_position: Position in the sequence
            total_sequences: Total number of sequences

        Returns:
            Dictionary with clip information
        """
        if not self.video_clips:
            self.find_video_clips()

        # Initialize clip usage count if needed
        for clip in self.video_clips:
            if clip not in self.clip_usage_count:
                self.clip_usage_count[clip] = 0

        # Get available clips
        available_clips = self.video_clips.copy()

        # Apply smart selection logic
        use_smart_logic = self.config.get("avoid_clip_repetition", True)

        if use_smart_logic and available_clips:
            # Sort by usage count (least used first)
            available_clips.sort(key=lambda clip: self.clip_usage_count[clip])

            # Filter out clips that reached max repeats (if enabled)
            if self.config["avoid_clip_repetition"]:
                max_repeats = self.config["max_clip_repeats"]
                available_clips = [
                    clip
                    for clip in available_clips
                    if self.clip_usage_count[clip] < max_repeats
                ]

            # If no clips available after filtering, reset usage counts
            if not available_clips:
                self.logger.warning(
                    "No clips available after filtering, resetting usage counts"
                )
                self.clip_usage_count = {clip: 0 for clip in self.video_clips}
                available_clips = self.video_clips.copy()

        # Select a clip
        selected_clip = random.choice(available_clips)

        # Update usage count
        self.clip_usage_count[selected_clip] += 1

        # Get clip duration
        duration = get_video_duration(selected_clip)

        return {
            "path": selected_clip,
            "duration": duration,
            "usage_count": self.clip_usage_count[selected_clip],
        }

    def create_video(
        self, scenes: List[Dict[str, Any]], output_dir: Union[str, Path]
    ) -> Path:
        """Create a video from scenes.

        Args:
            scenes: List of scene dictionaries
            output_dir: Directory to save the output

        Returns:
            Path to the created video
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create a list file for FFmpeg
        list_file = self.temp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for i, scene in enumerate(scenes):
                # Select a clip
                clip_info = self._select_clip()

                # Trim the clip to the scene duration
                clip_path = Path(clip_info['path'])
                trimmed_path = (
                    self.temp_dir / f"trimmed_{i}_{clip_path.name}"
                )

                # Use build_trim_command for trimming operations
                cmd = build_trim_command(
                    input_path=clip_path,
                    output_path=trimmed_path,
                    start_time=0.0,  # Trim from the beginning
                    end_time=scene["duration"],  # Trim to the required scene duration
                    video_codec=self.config["video_codec"],
                    # Explicitly cast the preset value to string
                    preset=cast(str, self.QUALITY_PRESETS[self.config["quality"]]["preset"]),
                    # Use .get() for bitrate with a default and cast to string
                    bitrate=cast(str, self.QUALITY_PRESETS[self.config["quality"]].get("video_bitrate", "4M")),
                    fps=self.config.get("fps", 30),
                    pix_fmt=self.config["pix_fmt"]
                )

                run_ffmpeg_command(cmd)

                # Add to concat list
                f.write(f"file '{trimmed_path.absolute()}'\n")

        # Concatenate all trimmed clips
        output_path = output_dir / "final_video.mp4"

        # Use FFmpeg to concatenate clips
        cmd = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-c",
            "copy",
            "-y",
            str(output_path),
        ]

        run_ffmpeg_command(cmd)

        return output_path

    def combine_video_with_audio(
        self,
        video_path: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        volume: Optional[float] = None,
    ) -> Path:
        """Combine video with audio."""
        video_path = Path(video_path)
        audio_path = Path(audio_path)
        output_path = Path(output_path)

        # Build FFmpeg command
        filters = []
        if volume is not None:
            filters.append(f"volume={volume}")

        cmd = build_ffmpeg_command(
            input_paths=[video_path, audio_path],
            output_path=output_path,
            filters=filters,
            codec=self.config.get("video_codec", "libx264"),
            preset=self.config.get("preset", "medium"),
            crf=self.config.get("crf", 23),
            pix_fmt=self.config.get("pix_fmt", "yuv420p"),
        )

        # Run FFmpeg command
        run_ffmpeg_command(cmd)

        return output_path

    def concatenate_videos(
        self,
        video_paths: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        transition: Optional[str] = None,
        transition_duration: Optional[float] = None,
    ) -> Path:
        """Concatenate multiple videos."""
        # Convert all paths to Path objects
        video_paths_resolved = [Path(p) for p in video_paths]
        output_path = Path(output_path)

        # Build FFmpeg command
        filters = []
        if transition and transition_duration:
            filters.append(
                f"xfade=transition={transition}:duration={transition_duration}"
            )

        cmd = build_ffmpeg_command(
            input_paths=video_paths_resolved,
            output_path=output_path,
            filters=filters,
            codec=self.config.get("video_codec", "libx264"),
            preset=self.config.get("preset", "medium"),
            crf=self.config.get("crf", 23),
            pix_fmt=self.config.get("pix_fmt", "yuv420p"),
        )

        # Run FFmpeg command
        run_ffmpeg_command(cmd)

        return output_path

    def create_final_video(
        self,
        segments: List[Dict[str, Any]],
        output_path: Union[str, Path],
        add_fade: bool = False,
        fade_duration: float = 1.0,
    ) -> Path:
        """Create the final video by concatenating all segments."""
        output_path = Path(output_path)
        
        # Get quality settings safely
        quality_key = self.config.get("quality", "high")
        quality_settings = self.QUALITY_PRESETS.get(quality_key, self.QUALITY_PRESETS["high"])
        
        # Create a temporary file list for FFmpeg
        list_file = self.temp_dir / "video_list.txt"
        with open(list_file, "w") as f:
            for segment in segments:
                f.write(f"file '{segment['path']}'\n")
                if add_fade and segment != segments[-1]:
                    f.write(f"duration {segment['duration'] - fade_duration}\n")
                else:
                    f.write(f"duration {segment['duration']}\n")

        # Build FFmpeg command for concatenation
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", self.config.get("video_codec", "libx264"),
            "-preset", cast(str, quality_settings.get("preset", "medium")),
            "-crf", str(cast(int, quality_settings.get("crf", 23))),
            "-pix_fmt", self.config.get("pix_fmt", "yuv420p"),
            "-c:a", "aac",
            str(output_path)
        ]

        run_ffmpeg_command(cmd)
        return output_path
