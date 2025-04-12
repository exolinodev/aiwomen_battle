"""
EnhancedRhythmicVideoEditor for Women Around The World.

This module provides the EnhancedRhythmicVideoEditor class that extends the RhythmicVideoEditor
with additional capabilities for creating more sophisticated rhythmically synchronized videos.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import librosa

from watw.core.video.rhythmic import BeatInfo, RhythmicVideoEditor
from watw.core.video.utils import (
    build_ffmpeg_command,
    get_video_duration,
    get_video_info,
    run_ffmpeg_command,
)


class EnhancedRhythmicVideoEditor(RhythmicVideoEditor):
    """Enhanced video editor for creating sophisticated rhythmically synchronized videos."""

    # Available transition types
    TRANSITION_TYPES = {
        "none": "No transition (hard cut)",
        "crossfade": "Smooth crossfade between clips",
        "fade": "Fade to black and back",
        "wipe_left": "Wipe from left to right",
        "wipe_right": "Wipe from right to left",
        "zoom_in": "Zoom in transition",
        "zoom_out": "Zoom out transition",
        "random": "Random selection from available transitions",
    }

    # Available visual effects
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
        "random": "Random selection from available effects",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        temp_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize the EnhancedRhythmicVideoEditor.

        Args:
            config: Configuration parameters
            temp_dir: Directory for temporary files
        """
        super().__init__(config, temp_dir)

        # Additional configuration
        self.config.setdefault("transition_type", "random")
        self.config.setdefault("visual_effect", "random")
        self.config.setdefault("min_segment_duration", 0.5)
        self.config.setdefault("max_segment_duration", 5.0)
        self.config.setdefault("video_fadeout_duration", 0.0)
        self.config.setdefault("audio_fadeout_duration", 0.0)
        self.config.setdefault("audio_volume", 1.0)
        self.config.setdefault("fps", 30)
        self.config.setdefault("audio_codec", "aac")
        self.config.setdefault("audio_bitrate", "192k")

        # Initialize clip usage tracking
        self.clip_usage_count: Dict[Path, int] = {}
        self.clip_durations: Dict[Path, float] = {}

        # Initialize beat times
        self.beat_times: Optional[List[float]] = None
        self.effective_max_time: Optional[float] = None

    def find_video_clips(self, extensions: Optional[List[str]] = None) -> List[Path]:
        """Find all video clips in the specified directory.

        Args:
            extensions: List of video file extensions to search for

        Returns:
            List of paths to video clips
        """
        clips = super().find_video_clips(extensions)

        # Analyze clips for additional information
        self._analyze_clips()

        return clips

    def _analyze_clips(self) -> None:
        """Analyze video clips to determine their characteristics."""
        self.clip_analysis: Dict[str, Dict[str, Any]] = {}
        for clip_path in self.video_clips:
            try:
                info = get_video_info(clip_path)
                if info:
                    self.clip_analysis[str(clip_path)] = {
                        "duration": info.get("duration", 0.0),
                        "fps": info.get("fps", 30.0),
                        "resolution": info.get("resolution", (1920, 1080)),
                        "aspect_ratio": info.get("aspect_ratio", 16 / 9),
                    }
            except Exception as e:
                self.logger.warning(f"Error analyzing clip {clip_path}: {e}")

    def detect_beats(
        self,
        audio_path: Optional[Union[str, Path]] = None,
        sensitivity: float = 1.2,
        min_beats: int = 10,
        min_bpm: float = 60.0,
        max_bpm: float = 180.0,
        max_duration: Optional[float] = None,
    ) -> BeatInfo:
        """Detect beats in the audio file."""
        if not audio_path:
            raise ValueError("Audio path is required for beat detection")

        try:
            # Load audio file and detect beats
            y, sr = librosa.load(str(audio_path), duration=max_duration)
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

            # Convert beat frames to times
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=512)

            if len(beat_times) < min_beats:
                raise ValueError(
                    f"Not enough beats detected (found {len(beat_times)}, minimum {min_beats})"
                )

            if tempo < min_bpm or tempo > max_bpm:
                raise ValueError(
                    f"Detected tempo {tempo} BPM is outside allowed range ({min_bpm}-{max_bpm})"
                )

            return BeatInfo(
                tempo=float(tempo),
                beat_times=beat_times.tolist(),
                duration=float(len(y) / sr),
            )
        except Exception as e:
            self.logger.error(f"Error detecting beats: {e}")
            raise

    def create_beat_synchronized_video(
        self,
        output_path: Optional[Union[str, Path]] = None,
        add_fade: bool = True,
        fade_duration: float = 1.0,
    ) -> Path:
        """Create a video synchronized with beats.

        Args:
            output_path: Path to save the output
            add_fade: Whether to add a fade effect
            fade_duration: Duration of the fade in seconds

        Returns:
            Path to the final video
        """
        # Ensure beats are detected
        if self.beat_times is None:
            try:
                # Use self.audio_path which should be set during init or config
                self.detect_beats(audio_path=self.audio_path)  # Assuming audio path is available
            except Exception as e:
                self.logger.error(f"Failed to detect beats before creating segments: {e}")
                raise ValueError("Beat detection failed, cannot create synchronized video.") from e
            # Check again after attempting detection
            if self.beat_times is None:
                raise ValueError("self.beat_times is still None after detection attempt.")

        # Ensure video clips are found
        if not self.video_clips:
            self.find_video_clips()  # This also calls _analyze_clips

        # Now self.beat_times is guaranteed to be List[float]
        if len(self.beat_times) < 2:  # Need at least two beats for intervals
            raise ValueError("Not enough beats detected to create segments.")

        if not output_path:
            output_path = (
                Path(self.config.get("output_dir", ".")) / "enhanced_rhythmic_video.mp4"
            )
        else:
            output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create segments based on beat times
        segments = []
        previous_clip_path = None

        # Get configuration values
        min_segment_dur_config = self.config.get("min_segment_duration", 0.5)
        max_segment_dur_config = self.config.get("max_segment_duration", 5.0)
        min_allowed_segment_dur = 0.1  # Absolute minimum segment duration

        # Calculate total beat intervals
        total_beat_intervals = len(self.beat_times) - 1

        self.logger.info(
            f"Creating {total_beat_intervals} segments based on {len(self.beat_times)} beats..."
        )

        # --- Segment Creation Loop ---
        for i in range(total_beat_intervals):
            start_time = self.beat_times[i]
            end_time = self.beat_times[i + 1]
            target_duration = end_time - start_time

            # Skip segment if interval is too small (e.g., due to duplicate beats somehow)
            if target_duration <= 0.01:
                self.logger.warning(
                    f"Skipping segment {i + 1} due to near-zero target duration ({target_duration:.3f}s)."
                )
                continue

            self.logger.info(
                f"\nSegment {i + 1}/{total_beat_intervals} ({start_time:.2f}s -> {end_time:.2f}s)"
            )
            self.logger.info(f"  Target Duration: {target_duration:.2f}s")

            # Apply min/max segment duration constraints from config
            segment_duration = np.clip(
                target_duration, min_segment_dur_config, max_segment_dur_config
            )
            duration_changed = False
            if abs(segment_duration - target_duration) > 0.01:
                duration_changed = True

            if duration_changed:
                self.logger.info(
                    f"  Adjusted Duration (Min/Max): {segment_duration:.2f}s (Original target: {target_duration:.2f}s)"
                )

            # --- Select Clip & Handle Duration/Speed Constraints ---
            clip_info = self._select_clip(
                previous_clip=previous_clip_path,
                sequence_position=i,
                total_sequences=total_beat_intervals,
            )
            clip_path = clip_info["path"]
            speed = clip_info["speed"]
            visual_effect = clip_info["visual_effect"]
            is_reverse = clip_info["reverse"]

            # Calculate required source duration based on target segment duration and speed
            source_duration_needed = segment_duration * speed

            original_clip_duration = self.get_clip_duration(clip_path)
            if original_clip_duration is None:
                self.logger.error(
                    f"Cannot get duration for selected clip {os.path.basename(clip_path)}, skipping segment."
                )
                continue  # Skip this segment

            # --- Attempt to find a suitable clip (long enough) ---
            max_attempts = 3
            attempt = 0
            clip_found_suitable = False
            while attempt < max_attempts:
                # Check if the current clip is long enough
                if original_clip_duration >= source_duration_needed:
                    clip_found_suitable = True
                    break  # Found suitable clip

                attempt += 1
                self.logger.warning(
                    f"Clip {os.path.basename(clip_path)} ({original_clip_duration:.2f}s) too short for {source_duration_needed:.2f}s @ {speed:.2f}x. Re-selecting (Attempt {attempt}/{max_attempts})..."
                )

                # Try selecting a different clip
                new_clip_info = self._select_clip(
                    previous_clip=clip_path,  # Avoid immediate repeat
                    sequence_position=i,
                    total_sequences=total_beat_intervals,
                )

                # Update clip info and re-evaluate
                clip_info = new_clip_info
                clip_path = clip_info["path"]
                speed = clip_info["speed"]  # Re-evaluate speed for the new clip
                visual_effect = clip_info["visual_effect"]  # Re-evaluate effect
                is_reverse = clip_info["reverse"]  # Re-evaluate reverse

                original_clip_duration = self.get_clip_duration(clip_path)
                if original_clip_duration is None:
                    self.logger.error(
                        f"Error getting duration for new clip {os.path.basename(clip_path)}, stopping attempts for this segment."
                    )
                    break  # Exit while loop

                source_duration_needed = (
                    segment_duration * speed
                )  # Recalculate needed duration
                self.logger.info(
                    f"  Trying Clip: {os.path.basename(clip_path)} ({original_clip_duration:.2f}s). Need {source_duration_needed:.2f}s @ {speed:.2f}x"
                )

            # --- Handle case where no suitable clip was found ---
            if not clip_found_suitable:
                if (
                    original_clip_duration is None
                ):  # Failed to get duration during attempts
                    self.logger.error(
                        f"Could not verify duration for clips. Skipping segment {i + 1}."
                    )
                    continue

                # Clip is still too short after attempts, must adjust segment duration to fit this clip
                self.logger.warning(
                    f"No clip found long enough after {max_attempts} attempts. Forcing segment {i + 1} to fit {os.path.basename(clip_path)} ({original_clip_duration:.2f}s)."
                )
                source_duration_needed = original_clip_duration  # Use the full available duration of the clip
                segment_duration = (
                    source_duration_needed / speed
                )  # Recalculate the final duration of this segment based on the source and speed
                segment_duration = max(
                    min_allowed_segment_dur, segment_duration
                )  # Ensure it's not too short

                # Check against safe minimum again AFTER recalculation
                if segment_duration < min_allowed_segment_dur:
                    self.logger.error(
                        f"Shortened segment duration ({segment_duration:.2f}s) still below safe minimum ({min_allowed_segment_dur:.2f}s). Skipping segment."
                    )
                    continue
                self.logger.info(
                    f"  New Final Segment Duration: {segment_duration:.2f}s"
                )
                # Declare clip_start once before the if/else
                clip_start: float
                clip_start = 0.0  # Start from beginning since we are using the whole clip
            else:
                # Clip is long enough, choose random start point within the available range
                max_start = original_clip_duration - source_duration_needed
                clip_start = random.uniform(0, max_start) if max_start > 0 else 0.0

            # Update previous clip path for next iteration's selection logic
            previous_clip_path = clip_path

            self.logger.info(f"  Using Clip: {os.path.basename(clip_path)}")
            self.logger.info(f"  Speed: {speed:.2f}x")
            self.logger.info(f"  Effect: {visual_effect}")
            self.logger.info(f"  Reverse: {is_reverse}")
            self.logger.info(f"  Start: {clip_start:.2f}s")
            self.logger.info(f"  Duration: {segment_duration:.2f}s")

            # Create segment
            segment = {
                "clip_path": clip_path,
                "speed": speed,
                "visual_effect": visual_effect,
                "reverse": is_reverse,
                "duration": segment_duration,
                "start_time": clip_start,
            }

            segments.append(segment)

        # Create final video
        return self.create_final_video(segments, output_path, add_fade, fade_duration)

    def get_clip_duration(self, clip_path: Union[str, Path]) -> Optional[float]:
        """Get the duration of a clip.

        Args:
            clip_path: Path to the clip

        Returns:
            Duration in seconds or None if not available
        """
        clip_path = Path(clip_path)

        # Check if we already have the duration
        if clip_path in self.clip_durations:
            return self.clip_durations[clip_path]

        # Try to get the duration
        try:
            duration: float = get_video_duration(clip_path)
            self.clip_durations[clip_path] = duration
            return duration
        except Exception as e:
            self.logger.error(f"Error getting duration for clip {clip_path}: {e}")
            return None

    def _select_clip(
        self,
        previous_clip: Optional[Union[str, Path]] = None,
        sequence_position: int = 0,
        total_sequences: int = 1,
    ) -> Dict[str, Any]:
        """Select a clip for a segment with enhanced options.

        Args:
            previous_clip: Path to the previous clip
            sequence_position: Position in the sequence
            total_sequences: Total number of sequences

        Returns:
            Dictionary with clip information
        """
        clip_info = super()._select_clip(
            previous_clip, sequence_position, total_sequences
        )

        # Add enhanced features
        clip_info["speed"] = random.uniform(
            0.8, 1.2
        )  # Random speed between 0.8x and 1.2x
        clip_info["visual_effect"] = self._get_visual_effect()
        clip_info["reverse"] = random.random() < 0.3  # 30% chance of reversing

        return clip_info

    def _get_visual_effect(self) -> str:
        """Get a visual effect to apply to a clip.

        Returns:
            Name of the visual effect
        """
        effect = self.config.get("visual_effect", "random")
        if effect == "random" or effect not in self.VISUAL_EFFECTS:
            # Exclude 'random' from possible choices
            effect = random.choice(list(k for k in self.VISUAL_EFFECTS.keys() if k != 'random'))
        assert isinstance(effect, str), "Visual effect must be a string"
        return effect

    def _get_transition(self) -> str:
        """Get a transition type to use between clips.

        Returns:
            Name of the transition
        """
        transition = self.config.get("transition_type", "random")
        if transition == "random" or transition not in self.TRANSITION_TYPES:
            # Exclude 'random' from possible choices
            transition = random.choice(list(k for k in self.TRANSITION_TYPES.keys() if k != 'random'))
        assert isinstance(transition, str), "Transition must be a string"
        return transition

    def _apply_visual_effect(
        self, input_path: Union[str, Path], output_path: Union[str, Path], effect: str
    ) -> None:
        """Apply a visual effect to a video."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        # Build FFmpeg command with appropriate filter
        filters = []
        if effect == "grayscale":
            filters.append("hue=s=0")
        elif effect == "sepia":
            filters.append(
                "colorchannelmixer=.393:.769:.189:.349:.686:.168:.272:.534:.131"
            )
        elif effect == "vibrance":
            filters.append("eq=saturation=1.5")
        elif effect == "vignette":
            filters.append("vignette=angle=PI/4:x0=0.5:y0=0.5")
        elif effect == "blur_edges":
            filters.append("boxblur=20:20:enable='between(t,0,0.5)'")
        elif effect == "sharpen":
            filters.append("unsharp=5:5:1.0:5:5:0.0")
        elif effect == "mirror":
            filters.append("hflip")
        elif effect == "flip":
            filters.append("vflip")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vf", ",".join(filters),
            "-c:v", self.config.get("video_codec", "libx264"),
            "-preset", self.QUALITY_PRESETS[self.config["quality"]]["preset"],
            "-crf", str(self.QUALITY_PRESETS[self.config["quality"]]["crf"]),
            "-pix_fmt", self.config.get("pix_fmt", "yuv420p"),
            "-c:a", "copy",
            str(output_path),
        ]

        # Run FFmpeg command
        run_ffmpeg_command(cmd)

    def _create_transition(
        self,
        clip1: Union[str, Path],
        clip2: Union[str, Path],
        output_path: Union[str, Path],
        transition_type: str,
        duration: float,
    ) -> None:
        """Create a transition between two video clips."""
        clip1 = Path(clip1)
        clip2 = Path(clip2)
        output_path = Path(output_path)

        # Build FFmpeg command with transition filter
        filters = []
        if transition_type == "crossfade":
            filters.append(f"xfade=transition=fade:duration={duration}")
        elif transition_type == "wipe_left":
            filters.append(f"xfade=transition=slideleft:duration={duration}")
        elif transition_type == "wipe_right":
            filters.append(f"xfade=transition=slideright:duration={duration}")
        elif transition_type == "zoom_in":
            filters.append(f"xfade=transition=zoom:duration={duration}")
        elif transition_type == "zoom_out":
            filters.append(f"xfade=transition=zoomin:duration={duration}")

        cmd = build_ffmpeg_command(
            input_paths=[clip1, clip2],
            output_path=output_path,
            filters=filters,
            codec=self.config.get("video_codec", "libx264"),
            preset=self.config.get("preset", "medium"),
            crf=self.config.get("crf", 23),
            pix_fmt=self.config.get("pix_fmt", "yuv420p"),
        )

        # Run FFmpeg command
        run_ffmpeg_command(cmd)

    def create_final_video(
        self,
        segments: List[Dict[str, Any]],
        output_path: Union[str, Path],
        add_fade: bool = False,
        fade_duration: float = 1.0,
    ) -> Path:
        """Create the final video from segments with enhanced features.

        Args:
            segments: List of segment dictionaries
            output_path: Path to save the output
            add_fade: Whether to add a fade effect
            fade_duration: Duration of the fade

        Returns:
            Path to the final video
        """
        output_path = Path(output_path)

        # Process each segment
        processed_segments: List[Path] = []

        for i, segment in enumerate(segments):
            # Select a clip with enhanced features
            clip_info = self._select_clip()

            # Create a temporary file for the processed segment
            temp_segment = self.temp_dir / f"segment_{i}.mp4"

            # Apply visual effect
            self._apply_visual_effect(
                clip_info["path"], temp_segment, clip_info["visual_effect"]
            )

            # Apply speed and reverse if needed
            if clip_info["speed"] != 1.0 or clip_info["reverse"]:
                temp_processed = self.temp_dir / f"segment_{i}_processed.mp4"

                filters = []
                if clip_info["speed"] != 1.0:
                    filters.append(f"setpts={1 / clip_info['speed']}*PTS")
                if clip_info["reverse"]:
                    filters.append("reverse")

                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_segment),
                    "-vf", ",".join(filters),
                    "-c:v", self.config.get("video_codec", "libx264"),
                    "-preset", self.QUALITY_PRESETS[self.config["quality"]]["preset"],
                    "-crf", str(self.QUALITY_PRESETS[self.config["quality"]]["crf"]),
                    "-pix_fmt", self.config.get("pix_fmt", "yuv420p"),
                    "-c:a", "copy",
                    str(temp_processed),
                ]

                run_ffmpeg_command(cmd)
                temp_segment.unlink()
                temp_segment = temp_processed

            # Trim to the desired duration
            if segment["duration"] > 0:
                temp_trimmed = self.temp_dir / f"segment_{i}_trimmed.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(temp_segment),
                    "-ss", "0",
                    "-t", str(segment["duration"]),
                    "-c:v", "copy",
                    "-c:a", "copy",
                    str(temp_trimmed),
                ]

                run_ffmpeg_command(cmd)
                temp_segment.unlink()
                temp_segment = temp_trimmed

            processed_segments.append(temp_segment)

        # Create transitions between segments
        if len(processed_segments) > 1:
            transition_outputs: List[Path] = []
            for i in range(len(processed_segments) - 1):
                transition_output = self.temp_dir / f"transition_{i}.mp4"
                transition_type = self._get_transition()
                transition_duration = self.config.get("transition_duration", 0.5)

                self._create_transition(
                    processed_segments[i],
                    processed_segments[i + 1],
                    transition_output,
                    transition_type,
                    transition_duration,
                )

                transition_outputs.append(transition_output)

            # Clean up processed segments
            for segment in processed_segments:
                if isinstance(segment, Path):
                    segment.unlink()
                elif isinstance(segment, dict) and 'path' in segment:
                    Path(segment['path']).unlink()

            # Concatenate all transitions
            final_output: Path = self.temp_dir / "final_transitions.mp4"
            self.concatenate_videos(transition_outputs, final_output)

            # Clean up transition outputs
            for output in transition_outputs:
                output.unlink()

            # Add fade if requested
            if add_fade:
                fade_output: Path = self.temp_dir / "final_fade.mp4"
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(final_output),
                    "-vf", f"fade=t=in:st=0:d={fade_duration}",
                    "-vf", f"fade=t=out:st={get_video_duration(final_output) - fade_duration}:d={fade_duration}",
                    "-c:v", "copy",
                    "-c:a", "copy",
                    str(fade_output),
                ]

                run_ffmpeg_command(cmd)
                final_output.unlink()
                final_output = fade_output

            # Copy to final output
            import shutil

            shutil.copy2(final_output, output_path)
            final_output.unlink()
        else:
            # Just one segment, copy it directly
            import shutil

            shutil.copy2(processed_segments[0], output_path)
            processed_segments[0].unlink()

        return output_path
