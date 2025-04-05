#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import argparse
import random
from glob import glob
import librosa
# Ensure you have scipy installed: pip install scipy
try:
    from scipy.signal import find_peaks
except ImportError:
    print("Scipy not found. Please install it: pip install scipy")
    find_peaks = None # Will cause an error later if needed, but allows script to load
import shutil
import sys
import json
from collections import defaultdict
from datetime import datetime

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
        "random": "Random selection from available transitions" # This is handled by logic, not an FFmpeg filter
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
        "random": "Random selection from available effects" # Handled by logic
    }

    def __init__(self, clips_dir, audio_path, output_file=None, temp_dir=None, config=None):
        """Initialize the editor with paths and configuration"""
        if find_peaks is None:
            raise ImportError("Scipy is required for beat detection. Please install it: pip install scipy")

        self.clips_dir = clips_dir
        self.audio_path = audio_path

        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if output_file is None:
            # Default output in the *parent* directory of the clips dir
            clips_parent_dir = os.path.dirname(clips_dir) if os.path.dirname(clips_dir) else "."
            output_file = os.path.join(clips_parent_dir, f"rhythmic_video_{self.timestamp}.mp4")
        self.output_file = output_file

        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(output_file), f"temp_{self.timestamp}")
        self.temp_dir = temp_dir

        # Default configuration
        self.default_config = {
            # General
            "duration": 60.0, # Maximum output duration (seconds). Default to 60s for Shorts.
            "resolution": "1080x1920", # YouTube Shorts resolution
            "fps": 30, # Standard frame rate for Shorts

            # Beat detection parameters
            "beat_sensitivity": 1.0,
            "min_beats": 20,
            "beat_offset": 0.0,
            "beat_grouping": 1,

            # Clip selection parameters
            "min_clip_duration": 0.5, # Min segment length based on beats
            "max_clip_duration": 5.0, # Max segment length based on beats
            "clip_speed_variation": True,
            "min_speed": 0.5,
            "max_speed": 2.0,
            "speed_change_probability": 0.3,
            "reverse_clip_probability": 0.15,

            # Transition parameters
            "transition_type": "crossfade",
            "transition_duration": 0.5,
            "hard_cut_ratio": 0.3,
            "transition_variation": True,
            "transition_safe": False,
            "fallback_transition": "crossfade",

            # Visual parameters
            "quality": "high", # Affects encoding CRF/preset
            "apply_visual_effects": False,
            "visual_effect_probability": 0.2,

            # Rhythm parameters (mostly implicit via beat detection)
            "use_musical_structure": True,

            # Advanced parameters
            "smart_clip_selection": True,
            "avoid_clip_repetition": True,
            "max_clip_repeats": 2,

            # Randomization Parameters
            "randomize_clip_selection_strength": 0.8,
            "randomize_speed_logic": False,
            "force_random_effect": False,
            "randomize_transition_selection": False,

            # Output parameters
            "audio_fade_in": 1.0,
            "audio_fade_out": 3.0,
            "video_fade_out_duration": 0.0,

            # YouTube Shorts specific
            "video_codec": "libx264",
            "audio_codec": "aac",
            "audio_bitrate": "384k",
            "video_bitrate": "8M", # 8 Mbps for SDR videos
            "pix_fmt": "yuv420p", # Required for compatibility
        }

        # Override defaults with provided configuration
        self.config = self.default_config.copy()
        if config:
            # Allow new keys beyond the defaults
            self.config.update(config)

        # Ensure the temporary directory exists and is empty
        if os.path.exists(self.temp_dir):
            print(f"Temporary directory '{self.temp_dir}' exists. Removing it.")
            try:
                shutil.rmtree(self.temp_dir)
            except OSError as e:
                raise OSError(f"Failed to remove existing temporary directory '{self.temp_dir}': {e}") from e
        try:
            os.makedirs(self.temp_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Failed to create temporary directory '{self.temp_dir}': {e}") from e


        # Instance variables
        self.video_clips = []
        self.beat_times = []
        self.audio_duration = None
        self.effective_max_time = None # Actual time limit used based on audio and config["duration"]
        self.clip_durations = {}
        self.clip_usage_count = defaultdict(int)
        self.clip_speeds = {}

    # --- find_video_clips, _analyze_clips, get_clip_duration ---
    def find_video_clips(self, extensions=None):
        """Find all video clips in the specified directory"""
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']

        self.video_clips = []
        print(f"Searching for video clips with extensions {extensions} in '{self.clips_dir}'...")
        for ext in extensions:
            # Case-insensitive search using glob patterns for both lower and upper case
            pattern_lower = os.path.join(self.clips_dir, f'*{ext.lower()}')
            pattern_upper = os.path.join(self.clips_dir, f'*{ext.upper()}')
            found_lower = glob(pattern_lower)
            found_upper = glob(pattern_upper)
            self.video_clips.extend(found_lower)
            # Only add upper case results if they are different files (e.g. case-sensitive filesystem)
            self.video_clips.extend([f for f in found_upper if f not in found_lower])


        # Remove duplicates and sort
        self.video_clips = sorted(list(set(self.video_clips)))

        if not self.video_clips:
            # Be more specific about why no clips were found
            if not os.path.isdir(self.clips_dir):
                 raise FileNotFoundError(f"Input directory does not exist: {self.clips_dir}")
            else:
                 raise FileNotFoundError(f"No video clips found in directory '{self.clips_dir}' with extensions {extensions}")

        print(f"Found {len(self.video_clips)} potential video clips.")
        if len(self.video_clips) > 20:
            print("Listing first 10:")
            for i, clip in enumerate(self.video_clips[:10]):
                 print(f" - {os.path.basename(clip)}")
            print(f" - ... and {len(self.video_clips) - 10} more.")
        else:
             for clip in self.video_clips:
                 print(f" - {os.path.basename(clip)}")


        self._analyze_clips() # Analyze durations after finding
        return self.video_clips

    def _analyze_clips(self):
        """Analyze all clips to gather metadata (duration) and filter invalid ones"""
        print("\nAnalyzing video clips for duration and validity...")
        original_count = len(self.video_clips)
        valid_clips = []
        removed_clips_info = [] # Store reasons for removal

        for i, clip_path in enumerate(self.video_clips):
            # Show progress less frequently for many clips
            if i % 20 == 0 or i == original_count - 1:
                 # Use carriage return to keep output on one line for progress
                 print(f"\rAnalyzing clip {i+1}/{original_count}: {os.path.basename(clip_path):<50}", end="")
                 sys.stdout.flush() # Ensure it prints immediately

            duration = self.get_clip_duration(clip_path) # Handles caching and ffprobe call

            if duration is None:
                 reason = "Duration analysis error"
                 removed_clips_info.append(f"{os.path.basename(clip_path)} ({reason})")
                 continue # Skip adding to durations cache and valid list
            elif duration <= 0.1: # Also check for too short clips
                 reason = f"Duration too short ({duration:.2f}s)"
                 removed_clips_info.append(f"{os.path.basename(clip_path)} ({reason})")
                 continue # Skip adding

            # If duration is valid, add to cache (done inside get_clip_duration) and list
            valid_clips.append(clip_path)

        print() # Newline after progress indicator

        self.video_clips = valid_clips
        removed_count = original_count - len(self.video_clips)

        if removed_count > 0:
             print(f"Removed {removed_count} clips due to errors or short duration:")
             # Print details of removed clips (limit if many)
             for i, info in enumerate(removed_clips_info):
                  if i < 5:
                      print(f"  - {info}")
                  elif i == 5:
                      print(f"  - ... and {removed_count - 5} more.")
                      break

        if not self.video_clips:
            raise ValueError("No valid video clips found after analysis. Check clip files and formats.")

        print(f"Clip analysis complete. Usable clips: {len(self.video_clips)}")


    def get_clip_duration(self, clip_path):
        """Get the duration of a clip using ffprobe, with caching and error handling."""
        # Return cached duration if available and valid
        if clip_path in self.clip_durations:
            # Check if cached value indicates a past failure
            if self.clip_durations[clip_path] is None:
                 return None
            return self.clip_durations[clip_path]

        # Use ffprobe to get duration
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration", # Get duration
            "-show_entries", "stream=codec_type", # Check for video stream
            "-of", "json", # Use JSON output for easier parsing
            clip_path
        ]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=20)
            metadata = json.loads(result.stdout)

            # Check if there's a video stream
            has_video = any(stream.get('codec_type') == 'video' for stream in metadata.get('streams', []))
            if not has_video:
                 print(f"Warning: Clip {os.path.basename(clip_path)} has no video stream.")
                 self.clip_durations[clip_path] = None # Cache failure
                 return None

            # Check for duration in format section
            duration_str = metadata.get('format', {}).get('duration')
            if duration_str is None:
                 # Try getting duration from the video stream if format duration is missing
                 video_stream = next((s for s in metadata.get('streams', []) if s.get('codec_type') == 'video'), None)
                 if video_stream and 'duration' in video_stream:
                     duration_str = video_stream['duration']
                     print(f"Note: Using video stream duration for {os.path.basename(clip_path)}.")
                 else:
                     print(f"Warning: Could not extract duration for {os.path.basename(clip_path)} from ffprobe JSON (format or stream).")
                     self.clip_durations[clip_path] = None
                     return None

            duration = float(duration_str)
            if duration <= 0:
                 print(f"Warning: Clip {os.path.basename(clip_path)} reported non-positive duration ({duration}).")
                 self.clip_durations[clip_path] = None # Cache failure (or treat 0 as failure)
                 return None

            self.clip_durations[clip_path] = duration
            return duration

        except subprocess.CalledProcessError as e:
            err_output = e.stderr.strip() if e.stderr else "(no stderr)"
            print(f"Warning: ffprobe error for {os.path.basename(clip_path)}. Error: {err_output}")
            self.clip_durations[clip_path] = None
            return None
        except subprocess.TimeoutExpired:
            print(f"Warning: ffprobe timed out analyzing {os.path.basename(clip_path)}.")
            self.clip_durations[clip_path] = None
            return None
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: Could not parse ffprobe output for {os.path.basename(clip_path)}. Error: {e}")
            self.clip_durations[clip_path] = None
            return None
        except Exception as e: # Catch any other unexpected errors
            print(f"Warning: Unexpected error getting duration for {os.path.basename(clip_path)}. Error: {e}")
            self.clip_durations[clip_path] = None
            return None

    # --- detect_beats ---
    def detect_beats(self):
        """
        Advanced beat detection, considering audio duration and user-defined max duration.
        """
        print(f"\nAnalyzing audio file: {os.path.basename(self.audio_path)}...")

        # Load audio and get duration
        try:
            y, sr = librosa.load(self.audio_path, sr=None) # Load with native sample rate
            self.audio_duration = librosa.get_duration(y=y, sr=sr)
            print(f"Audio duration: {self.audio_duration:.2f}s")
        except Exception as e:
            raise IOError(f"Could not load audio file {self.audio_path}: {e}")

        if self.audio_duration <= 0:
            raise ValueError("Audio file has zero or negative duration.")

        # Determine the maximum time limit for beats
        self.effective_max_time = self.audio_duration
        user_max_duration = self.config.get("duration") # Use .get for safety
        if user_max_duration is not None and user_max_duration > 0:
            if user_max_duration < self.effective_max_time:
                print(f"Applying user-specified maximum duration: {user_max_duration:.2f}s (from config/args)")
                self.effective_max_time = user_max_duration
            else:
                 print(f"User duration ({user_max_duration:.2f}s) is >= audio duration ({self.audio_duration:.2f}s). Using audio duration.")
        elif user_max_duration is not None: # User provided <= 0 value
            print(f"Warning: Invalid user duration specified ({user_max_duration}). Ignoring and using full audio duration.")

        print(f"Effective maximum time for video: {self.effective_max_time:.2f}s")

        # --- Beat Detection Logic ---
        sensitivity = self.config["beat_sensitivity"]
        min_beats_config = self.config["min_beats"]

        detected_beat_times = np.array([]) # Initialize

        # Perform actual beat detection using selected method
        if self.config["use_musical_structure"]:
            print("Using musical structure analysis (librosa.beat.beat_track)...")
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr,
                                                            trim=False, # Analyze full audio
                                                            units='frames')
                structure_beat_times = librosa.frames_to_time(beat_frames, sr=sr)

                # Combine with onsets for potentially more rhythmic points
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time', backtrack=True)

                # Combine, unique, sort
                all_beats = np.unique(np.concatenate([structure_beat_times, onset_times]))

                # Filter beats that are too close together
                min_beat_interval = 0.15 # Minimum seconds between distinct beats
                filtered_beats = []
                if len(all_beats) > 0:
                    last_beat = -min_beat_interval # Initialize to allow first beat
                    for beat in all_beats:
                        if beat - last_beat >= min_beat_interval:
                            filtered_beats.append(beat)
                            last_beat = beat
                detected_beat_times = np.array(filtered_beats)
                print(f"Initial combined beats found: {len(detected_beat_times)}")

            except Exception as e:
                print(f"Warning: Error during musical structure beat detection: {e}. Falling back to onset strength.")
                self.config["use_musical_structure"] = False # Force fallback for subsequent steps

        # Fallback or primary method: onset strength
        if not self.config["use_musical_structure"] or len(detected_beat_times) == 0: # Use onset if structure failed or yielded no beats
            print("Using onset strength analysis (librosa.onset.onset_strength)...")
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                # Use find_peaks with adjusted height based on sensitivity
                median_strength = np.median(onset_env[np.isfinite(onset_env)]) # Ensure finite values for median
                if not np.isfinite(median_strength): median_strength = 0.0 # Fallback if all are inf/nan
                # Sensitivity: 1.0 = median, > 1.0 more sensitive (lower threshold relative to median), < 1.0 less sensitive
                threshold_multiplier = 1.0 / max(0.1, sensitivity) # Avoid division by zero or extreme sensitivity
                threshold = median_strength * threshold_multiplier
                # Clamp threshold to reasonable percentiles to avoid detecting everything or nothing
                p10 = np.percentile(onset_env[np.isfinite(onset_env)], 10) if any(np.isfinite(onset_env)) else 0
                p90 = np.percentile(onset_env[np.isfinite(onset_env)], 90) if any(np.isfinite(onset_env)) else 0
                threshold = max(p10, min(threshold, p90))
                # Ensure threshold is not negative
                threshold = max(0, threshold)

                min_distance_samples = int(sr * 0.15) # Min 0.15s between peaks
                peaks, _ = find_peaks(onset_env, height=threshold, distance=min_distance_samples)
                detected_beat_times = librosa.frames_to_time(peaks, sr=sr)
                print(f"Onset peaks found: {len(detected_beat_times)}")
            except Exception as e:
                 raise ValueError(f"Error during onset strength beat detection: {e}") from e


        # Apply sensitivity scaling if using musical structure (crude method, refine?)
        if self.config["use_musical_structure"] and sensitivity != 1.0 and len(detected_beat_times) > 0:
             print(f"Applying sensitivity factor: {sensitivity} (experimental)")
             current_num_beats = len(detected_beat_times)
             target_num_beats = int(current_num_beats * sensitivity)
             target_num_beats = max(1, target_num_beats) # Ensure at least 1

             if target_num_beats > current_num_beats: # Increase beats (add midpoints probabilistically)
                 new_beats = []
                 prob_add = (sensitivity - 1.0) / max(1.0, sensitivity) # Probability scaled by how much > 1 sensitivity is
                 for i in range(current_num_beats - 1):
                     new_beats.append(detected_beat_times[i])
                     midpoint = (detected_beat_times[i] + detected_beat_times[i+1]) / 2
                     # Only add if interval isn't already tiny
                     if (detected_beat_times[i+1] - detected_beat_times[i] > 0.1) and random.random() < prob_add:
                         new_beats.append(midpoint)
                 new_beats.append(detected_beat_times[-1]) # Add last original beat
                 detected_beat_times = np.sort(np.array(new_beats))
                 print(f"  -> Increased beats to approx {len(detected_beat_times)}")

             elif target_num_beats < current_num_beats: # Decrease beats (random sampling)
                 indices = sorted(random.sample(range(current_num_beats), target_num_beats))
                 detected_beat_times = detected_beat_times[indices]
                 print(f"  -> Reduced beats to approx {len(detected_beat_times)}")


        # Apply beat grouping
        grouping = self.config["beat_grouping"]
        if grouping > 1 and len(detected_beat_times) > 0:
            print(f"Grouping beats by {grouping}...")
            detected_beat_times = detected_beat_times[::grouping]

        # Apply beat offset
        offset = self.config["beat_offset"]
        if offset != 0:
            print(f"Applying beat offset: {offset}s")
            detected_beat_times = detected_beat_times + offset
            # Clamp to 0 and audio duration initially
            detected_beat_times = detected_beat_times[detected_beat_times >= 0]


        # --- Filter beats by effective_max_time ---
        print(f"Filtering beats to conform to effective max time: {self.effective_max_time:.2f}s")
        beat_times_within_limit = detected_beat_times[detected_beat_times <= self.effective_max_time]


        # Ensure minimum number of beats *within the limit*
        if len(beat_times_within_limit) < min_beats_config:
            print(f"Warning: Detected only {len(beat_times_within_limit)} beats within the time limit ({self.effective_max_time:.2f}s) (min configured: {min_beats_config}).")
            if len(beat_times_within_limit) < 2: # Need at least 2 beats to make segments
                 print(f"Generating {min_beats_config} fallback regular beats within the time limit.")
                 num_fallback_beats = max(2, min_beats_config)
                 # Create beats evenly spaced between 0 and effective_max_time
                 # np.linspace includes start/end, so N+1 points give N intervals.
                 # We want the points defining the *ends* of segments starting from 0.
                 beat_times_within_limit = np.linspace(0, self.effective_max_time, num_fallback_beats + 1)[1:] # Points from first interval end to final end
                 print(f"  Fallback beats: {np.round(beat_times_within_limit, 2)}")


        # Ensure beats exist at or very near time 0 and effective_max_time
        final_beat_list = []
        # Add 0.0 if no beats or first beat is significantly after start
        if len(beat_times_within_limit) == 0 or beat_times_within_limit[0] > 0.05:
             final_beat_list.append(0.0)
        # Add the detected/generated beats within the limit
        if len(beat_times_within_limit) > 0:
             final_beat_list.extend(beat_times_within_limit)
        # Add effective_max_time if it's not already the last beat (or very close)
        if len(final_beat_list) == 0 or final_beat_list[-1] < self.effective_max_time - 0.05:
             # Avoid adding if effective_max_time is 0
             if self.effective_max_time > 0:
                 final_beat_list.append(self.effective_max_time)

        # Remove duplicates (e.g., if 0 or end time were already present) and sort
        self.beat_times = np.unique(np.array(final_beat_list))

        # Final check - need at least two points (start and end) to define segments
        if len(self.beat_times) < 2:
             # This might happen if effective_max_time is very small or zero
             print(f"Warning: Could not establish sufficient beat timings (only {len(self.beat_times)}). Adding 0 and effective max time.")
             self.beat_times = np.unique([0.0, max(0.01, self.effective_max_time)]) # Force start and end (ensure end > 0)
             if len(self.beat_times) < 2:
                  raise ValueError(f"Could not establish at least two beat timings (start/end) for effective duration {self.effective_max_time:.2f}s.")


        print(f"Using {len(self.beat_times)} final beat times between {self.beat_times[0]:.2f}s and {self.beat_times[-1]:.2f}s.")
        if len(self.beat_times) > 10:
            print(f"  First 5 beats: {np.round(self.beat_times[:5], 2)}")
            print(f"  Last 5 beats : {np.round(self.beat_times[-5:], 2)}")
        else:
            print(f"  Beat times   : {np.round(self.beat_times, 2)}")

        # Calculate intervals for info
        intervals = np.diff(self.beat_times)
        if len(intervals) > 0:
             print(f"  Segment intervals range from {np.min(intervals):.2f}s to {np.max(intervals):.2f}s (avg: {np.mean(intervals):.2f}s)")
        else:
             print("  Warning: Only one beat time found, cannot calculate intervals.")


        return self.beat_times

    # --- _select_clip, _select_transition, _apply_visual_effect, _apply_speed_effect, _create_transition, _create_hard_cut ---
    def _select_clip(self, previous_clip=None, sequence_position=0, total_sequences=1):
        """
        Intelligently or randomly select a clip based on configuration settings.
        """
        available_clips = self.video_clips.copy()
        if not available_clips:
            raise ValueError("No video clips available for selection!")

        selected_clip_path = None

        # Apply smart clip selection logic (or bypass based on randomness strength)
        use_smart_logic = self.config["smart_clip_selection"] and \
                          random.random() < self.config["randomize_clip_selection_strength"]

        if use_smart_logic and available_clips:
            # Sort by usage count (least used first)
            available_clips.sort(key=lambda clip: self.clip_usage_count[clip])

            # Filter out clips that reached max repeats (if enabled)
            if self.config["avoid_clip_repetition"]:
                max_repeats = self.config["max_clip_repeats"]
                filtered_clips = [clip for clip in available_clips
                                  if self.clip_usage_count[clip] < max_repeats]

                if not filtered_clips and available_clips:
                    print(f"  Info: All available clips used >= {max_repeats} times. Selecting least used overall.")
                elif filtered_clips:
                     available_clips = filtered_clips # Use the filtered list

            # Avoid repeating the immediate previous clip if possible and more than one option exists
            if previous_clip is not None and previous_clip in available_clips and len(available_clips) > 1:
                 temp_list = [c for c in available_clips if c != previous_clip]
                 if temp_list:
                     available_clips = temp_list

            # Select from the potentially filtered/sorted list
            if available_clips:
                 # Prefer less used clips using weights (simple linear decay for less bias)
                 weights = np.linspace(1.0, 0.1, len(available_clips))
                 weights = np.maximum(0.01, weights) # Ensure non-zero
                 weights /= weights.sum() # Normalize
                 try:
                     selected_clip_path = random.choices(available_clips, weights=weights, k=1)[0]
                 except (IndexError, ValueError) as e:
                     print(f"  Warning: Clip weighting selection failed ({e}), choosing first available.")
                     selected_clip_path = available_clips[0]
            else:
                print("  Warning: Smart selection resulted in no available clips. Falling back to random.")
                use_smart_logic = False


        # If smart logic was bypassed, failed, or no clips remained
        if not use_smart_logic or selected_clip_path is None:
             if not self.video_clips: # Final safety check
                  raise ValueError("Cannot select clip: No valid video clips exist.")
             selected_clip_path = random.choice(self.video_clips)


        # --- Determine Speed ---
        speed = 1.0
        if self.config["clip_speed_variation"] and random.random() < self.config["speed_change_probability"]:
            min_s = self.config["min_speed"]
            max_s = self.config["max_speed"]
            if self.config["randomize_speed_logic"]:
                speed = random.uniform(min_s, max_s)
            else:
                # Position-based logic
                position_ratio = sequence_position / max(1, total_sequences -1) if total_sequences > 1 else 0.5 # Ratio 0 to 1
                # Simple 3-part split
                if position_ratio < 0.33: speed = random.uniform(min_s, 1.0) # Start slower
                elif position_ratio > 0.66: speed = random.uniform(1.0, max_s) # End faster
                else: speed = random.uniform(min_s, max_s) # Middle - full range
        speed = max(0.1, speed) # Ensure speed is not zero or negative


        # --- Determine Visual Effect ---
        visual_effect = "none"
        apply_effect = False
        if self.config["apply_visual_effects"]: # Check base flag first
             if self.config["force_random_effect"]:
                  apply_effect = True
             elif random.random() < self.config["visual_effect_probability"]:
                  apply_effect = True

        if apply_effect:
             available_effects = [k for k in self.VISUAL_EFFECTS if k not in ["none", "random"]]
             if available_effects:
                 visual_effect = random.choice(available_effects)


        # --- Determine Reverse ---
        is_reverse = False
        clip_duration = self.clip_durations.get(selected_clip_path) # Use cached duration
        min_duration_for_reverse = 1.0 # Don't reverse very short clips
        if clip_duration is not None and clip_duration > min_duration_for_reverse:
             if random.random() < self.config["reverse_clip_probability"]:
                 is_reverse = True


        # Increment usage count for the selected clip
        self.clip_usage_count[selected_clip_path] += 1

        return {
            "path": selected_clip_path,
            "speed": speed,
            "visual_effect": visual_effect,
            "reverse": is_reverse
        }

    def _select_transition(self, segment_index, total_segments):
        """Select a transition type based on configuration."""
        # Hard cut check first
        if random.random() < self.config["hard_cut_ratio"]:
            return "none"

        valid_transitions = [k for k in self.TRANSITION_TYPES if k != "random"]
        if not valid_transitions: return "none" # Safety

        fallback_transition = self.config.get("fallback_transition", "crossfade")
        if fallback_transition not in self.TRANSITION_TYPES: fallback_transition = "crossfade" # Ensure fallback is valid

        # Safe mode check
        if self.config["transition_safe"]:
             safe_transitions = ["none", "crossfade", "fade"]
             safe_available = [t for t in valid_transitions if t in safe_transitions]
             if not safe_available: return fallback_transition

             if not self.config["transition_variation"] and \
                not self.config["randomize_transition_selection"] and \
                self.config["transition_type"] in safe_available:
                 return self.config["transition_type"]
             else:
                 # If variation is allowed even in safe mode, pick random safe one
                 return random.choice(safe_available)


        # If pure random selection is forced
        if self.config["randomize_transition_selection"]:
             # Exclude 'none' as hard_cut_ratio handles it
             selectable = [t for t in valid_transitions if t != "none"] or [fallback_transition] # Fallback if only 'none' was valid
             return random.choice(selectable)


        # If variation is disabled, use the configured type
        if not self.config["transition_variation"]:
             selected_type = self.config["transition_type"]
             if selected_type == "random": # Handle random selection here too
                 selectable = [t for t in valid_transitions if t != "none"] or [fallback_transition]
                 return random.choice(selectable)
             else:
                 return selected_type if selected_type in valid_transitions else fallback_transition


        # --- Variation enabled, positional weighting (optional, complex) ---
        # This positional weighting is complex and might not always give desired results.
        # A simpler approach could be just random.choice from valid_transitions (excluding 'none').
        # Keeping the weighted logic for now as it was in the original script.
        reference_order = ["none", "crossfade", "fade", "wipe_left", "wipe_right", "zoom_in", "zoom_out"]
        reference_map = {name: i for i, name in enumerate(reference_order)}

        base_weights_start = [0.5, 3, 2, 1, 1, 0.5, 0.5] # Favor fades/crossfades early
        base_weights_end   = [0.5, 1, 1, 2, 2, 2, 2]     # Favor wipes/zooms late
        base_weights_mid   = [0.5, 2, 2, 1.5, 1.5, 1.5, 1.5] # Mixed middle

        position_ratio = segment_index / max(1, total_segments -1) if total_segments > 1 else 0.5 # Ratio 0 to 1

        if position_ratio < 0.25: chosen_weights_base = base_weights_start
        elif position_ratio > 0.75: chosen_weights_base = base_weights_end
        else: chosen_weights_base = base_weights_mid

        final_weights = []
        present_transitions = []
        for t_name in valid_transitions:
             # Skip 'none' in weighted selection as hard_cut_ratio handles it
             if t_name == "none": continue

             ref_index = reference_map.get(t_name)
             weight = 1.0 # Default weight for unmapped transitions
             if ref_index is not None and ref_index < len(chosen_weights_base):
                  weight = chosen_weights_base[ref_index]

             final_weights.append(max(0.01, weight)) # Ensure positive weight
             present_transitions.append(t_name)


        if not present_transitions: # Should only happen if valid_transitions only contained 'none'
             return "none"
        if sum(final_weights) <= 0: # All weights somehow became zero
             return random.choice(present_transitions) # Fallback to simple random choice

        try:
            return random.choices(present_transitions, weights=final_weights, k=1)[0]
        except ValueError as e:
             print(f"Warning: Issue selecting transition with weights ({e}), choosing randomly from available.")
             return random.choice(present_transitions)

    def _apply_visual_effect(self, input_file, output_file, effect):
        """Apply a visual effect to a clip using FFmpeg"""
        if not (os.path.exists(input_file) and os.path.getsize(input_file) > 0):
             print(f"  Error: Input file for visual effect '{effect}' is missing or empty: {os.path.basename(input_file)}")
             return None

        effect_filters = {
            "grayscale": "hue=s=0",
            "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
            "vibrance": "eq=saturation=1.5:contrast=1.1",
            "vignette": f"vignette=angle=PI/5:aspect={self.config['resolution'].replace('x', '/')}", # Use target aspect ratio
            "blur_edges": f"boxblur=luma_radius=5:luma_power=1, vignette=angle=PI/4:aspect={self.config['resolution'].replace('x', '/')}", # Simpler blur edge
            "sharpen": "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=0.8", # Milder sharpen
            "mirror": "hflip",
            "flip": "vflip"
        }

        filter_str = effect_filters.get(effect)

        if filter_str:
             # Determine quality settings for intermediate encoding
             quality = self.config["quality"]
             preset_cfg = self._get_quality_preset(quality)
             # Use slightly better CRF for intermediate files to minimize quality loss
             intermediate_crf = max(16, preset_cfg["crf"] - 2)

             cmd = [
                 "ffmpeg", "-loglevel", "warning", "-i", input_file,
                 "-vf", filter_str,
                 "-c:v", "libx264", "-preset", preset_cfg["preset"], "-crf", str(intermediate_crf),
                 "-an", output_file, "-y"
             ]
             try:
                 subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
                 if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return output_file
                 else: print(f"  Warning: Effect '{effect}' command ran but produced invalid output."); return None
             except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                 err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                 print(f"  Warning: Failed to apply effect '{effect}': {err_msg[-200:]}. Skipping effect.") # Show last part of stderr
                 # Fallback: Copy original
                 try: shutil.copy2(input_file, output_file); return output_file # Use copy2 to preserve metadata if possible
                 except Exception as copy_e: print(f"  Error: Could not copy original after effect failure: {copy_e}"); return None
        else:
            # No effect or unknown, just copy
            try: shutil.copy2(input_file, output_file); return output_file
            except Exception as copy_e: print(f"  Error: Could not copy file for 'none' effect: {copy_e}"); return None

    def _apply_speed_effect(self, input_file, output_file, speed):
        """Apply speed adjustment to a clip using FFmpeg"""
        if not (os.path.exists(input_file) and os.path.getsize(input_file) > 0):
             print(f"  Error: Input file for speed effect ({speed:.2f}x) is missing or empty: {os.path.basename(input_file)}")
             return None

        if abs(speed - 1.0) < 0.01: # Effectively 1.0x speed
            try: shutil.copy2(input_file, output_file); return output_file
            except Exception as copy_e: print(f"  Error: Could not copy file for 1.0x speed: {copy_e}"); return None

        pts_factor = 1.0 / speed
        pts_factor = max(0.01, min(100.0, pts_factor)) # Limit speed 0.01x - 100x
        filter_str = f"setpts={pts_factor:.4f}*PTS"

        # Determine quality settings
        quality = self.config["quality"]
        preset_cfg = self._get_quality_preset(quality)
        intermediate_crf = max(16, preset_cfg["crf"] - 2) # Better intermediate CRF

        cmd = [
            "ffmpeg", "-loglevel", "warning", "-i", input_file,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", preset_cfg["preset"], "-crf", str(intermediate_crf),
            "-an", output_file, "-y"
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60) # Longer timeout for speed change
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return output_file
            else: print(f"  Warning: Speed effect ({speed:.2f}x) command ran but produced invalid output."); return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"  Warning: Failed to apply speed effect ({speed:.2f}x): {err_msg[-200:]}. Skipping speed change.")
            # Fallback: Copy original
            try: shutil.copy2(input_file, output_file); return output_file
            except Exception as copy_e: print(f"  Error: Could not copy original after speed failure: {copy_e}"); return None

    def _create_transition(self, clip1, clip2, output_file, transition_type, duration):
        """Create a transition between two clips with improved error handling"""
        # --- Validation ---
        clip1_valid = os.path.exists(clip1) and os.path.getsize(clip1) > 0
        clip2_valid = os.path.exists(clip2) and os.path.getsize(clip2) > 0

        if not clip1_valid and not clip2_valid:
             print(f"  Error: Both input clips for transition invalid: {os.path.basename(clip1)}, {os.path.basename(clip2)}")
             return None
        if not clip1_valid:
             print(f"  Warning: Input clip1 invalid for transition. Using clip2 only.")
             try: shutil.copy2(clip2, output_file); return output_file
             except Exception as e: print(f"  Error copying clip2 during fallback: {e}"); return None
        if not clip2_valid:
             print(f"  Warning: Input clip2 invalid for transition. Using clip1 only.")
             try: shutil.copy2(clip1, output_file); return output_file
             except Exception as e: print(f"  Error copying clip1 during fallback: {e}"); return None

        # --- Duration Adjustment ---
        clip1_duration = None # Initialize
        try:
             clip1_duration = self.get_clip_duration(clip1)
             clip2_duration = self.get_clip_duration(clip2)
             if clip1_duration is None or clip1_duration <= 0 or clip2_duration is None or clip2_duration <= 0:
                  raise ValueError("Invalid clip durations obtained for transition.")

             min_safe_duration = 0.1 # Minimum allowed transition duration
             required_clip_time = duration # Time needed from end of clip1 and start of clip2 for the transition

             # Check if clips are long enough for the *entire* transition duration
             if clip1_duration < required_clip_time or clip2_duration < required_clip_time:
                  print(f"  Warning: Clips too short ({clip1_duration:.2f}s, {clip2_duration:.2f}s) for {duration:.2f}s {transition_type} transition.")
                  # Adjust duration to be a percentage of the shorter clip's available overlap time, clamped to min_safe_duration
                  adjusted_duration = max(min_safe_duration, min(clip1_duration, clip2_duration) * 0.8) # Use 80% of shorter clip?
                  # Ensure adjusted duration is not longer than original requested duration
                  adjusted_duration = min(duration, adjusted_duration)

                  print(f"  Adjusted transition duration to {adjusted_duration:.2f}s")
                  if adjusted_duration < min_safe_duration:
                       print(f"  Warning: Adjusted duration ({adjusted_duration:.2f}s) too short. Falling back to hard cut.")
                       return self._create_hard_cut(clip1, clip2, output_file)
                  duration = adjusted_duration # Update the duration to be used

        except ValueError as e:
             print(f"  Warning: {e}. Cannot perform transition. Falling back to hard cut.")
             return self._create_hard_cut(clip1, clip2, output_file)
        except Exception as e:
             print(f"  Warning: Duration check/adjustment failed for transition '{transition_type}': {e}. Using original {duration:.2f}s (may fail).")


        # --- FFmpeg Command ---
        # Base xfade filters (duration updated)
        transition_filters_base = {
            "crossfade" : f"xfade=transition=fade:duration={duration:.3f}",
            "fade"      : f"xfade=transition=fadeblack:duration={duration:.3f}",
            "wipe_left" : f"xfade=transition=wipeleft:duration={duration:.3f}",
            "wipe_right": f"xfade=transition=wiperight:duration={duration:.3f}",
            "zoom_in"   : f"xfade=transition=zoomin:duration={duration:.3f}",
            "zoom_out"  : f"xfade=transition=zoomout:duration={duration:.3f}"
        }

        offset = 0 # Default offset
        if clip1_duration:
            # Offset is when clip2 starts relative to the start of clip1.
            # For xfade, it starts fading clip1 out 'duration' seconds before its end.
            offset = max(0, clip1_duration - duration)
        else:
             print(f"  Warning: Could not get clip1 duration for transition offset calculation. Using offset 0 (may look incorrect).")

        filter_base = transition_filters_base.get(transition_type)

        if not filter_base or transition_type == "none":
            return self._create_hard_cut(clip1, clip2, output_file)

        filter_str = f"{filter_base}:offset={offset:.3f}" # Add calculated offset

        # Quality settings
        quality = self.config["quality"]
        preset_cfg = self._get_quality_preset(quality)
        # Use a consistent CRF for transitions, maybe slightly lower quality than intermediates is okay
        transition_crf = preset_cfg["crf"]

        cmd = [
            "ffmpeg", "-loglevel", "warning",
            "-i", clip1, "-i", clip2,
            "-filter_complex", f"[0:v][1:v]{filter_str},format={self.config['pix_fmt']}[outv]", # Ensure pixel format consistency
            "-map", "[outv]",
            "-c:v", "libx264", "-preset", preset_cfg["preset"], "-crf", str(transition_crf),
            "-an", output_file, "-y"
        ]

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0: return output_file
            else: print(f"  Error: Transition '{transition_type}' command ran but output file is invalid."); raise subprocess.CalledProcessError(1, cmd, stderr="Output invalid")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"  Error creating '{transition_type}' transition: {err_msg[-200:]}. Trying fallback.")

            # Fallback 1: Configured fallback transition
            fallback_type = self.config.get("fallback_transition", "crossfade")
            if fallback_type not in self.TRANSITION_TYPES: fallback_type = "crossfade" # Validate fallback
            # Only try fallback if it's different from the failed one and not 'none'
            if transition_type != fallback_type and fallback_type != "none":
                print(f"  Trying fallback transition: '{fallback_type}'...")
                fallback_filter_base = transition_filters_base.get(fallback_type) # Use same duration/offset logic
                if fallback_filter_base:
                     fallback_filter_str = f"{fallback_filter_base}:offset={offset:.3f}"
                     fallback_cmd = [
                         "ffmpeg", "-loglevel", "warning",
                         "-i", clip1, "-i", clip2,
                         "-filter_complex", f"[0:v][1:v]{fallback_filter_str},format={self.config['pix_fmt']}[outv]",
                         "-map", "[outv]", "-c:v", "libx264", "-preset", "fast", "-crf", str(transition_crf + 1), # Faster fallback preset
                         "-an", output_file, "-y"
                     ]
                     try:
                         subprocess.run(fallback_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45)
                         if os.path.exists(output_file) and os.path.getsize(output_file) > 0: print(f"  Fallback transition '{fallback_type}' successful."); return output_file
                         else: print("  Fallback transition command ran but output file is invalid."); raise subprocess.CalledProcessError(1, fallback_cmd, stderr="Output invalid")
                     except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e2:
                         err_msg2 = e2.stderr.decode(errors='ignore').strip() if hasattr(e2, 'stderr') and e2.stderr else str(e2)
                         print(f"  Fallback transition '{fallback_type}' failed: {err_msg2[-200:]}. Falling back to hard cut.")

            # Fallback 2: Hard cut
            print("  Falling back to hard cut.")
            return self._create_hard_cut(clip1, clip2, output_file)

    def _create_hard_cut(self, clip1, clip2, output_file):
        """Create a hard cut by concatenating two clips using FFmpeg concat demuxer."""
        concat_list_file = os.path.join(self.temp_dir, f"concat_{random.randint(1000,9999)}.txt")
        try:
            # Ensure clips exist before writing to list
            clip1_valid = os.path.exists(clip1) and os.path.getsize(clip1) > 0
            clip2_valid = os.path.exists(clip2) and os.path.getsize(clip2) > 0
            if not clip1_valid or not clip2_valid:
                 print(f"  Error (Hard Cut): Invalid inputs. Clip1 valid: {clip1_valid}, Clip2 valid: {clip2_valid}")
                 # Try to return the valid clip if one exists
                 if clip1_valid: shutil.copy2(clip1, output_file); return output_file
                 if clip2_valid: shutil.copy2(clip2, output_file); return output_file
                 return None

            # Escape paths for the concat file list
            safe_clip1 = os.path.abspath(clip1).replace("'", "'\\''")
            safe_clip2 = os.path.abspath(clip2).replace("'", "'\\''")
            with open(concat_list_file, 'w', encoding='utf-8') as f:
                f.write(f"file '{safe_clip1}'\n")
                f.write(f"file '{safe_clip2}'\n")

            # Attempt codec copy first (fastest if compatible)
            cmd_copy = [
                "ffmpeg", "-loglevel", "error", # Only show errors
                "-f", "concat", "-safe", "0", "-i", concat_list_file,
                "-c", "copy", output_file, "-y"
            ]
            process_copy = subprocess.run(cmd_copy, capture_output=True, text=True, timeout=30)
            if process_copy.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return output_file
            else:
                print(f"  Hard cut (copy codec) failed or produced empty file. Stderr: {process_copy.stderr.strip()[-200:]}. Trying re-encoding.")
                if os.path.exists(output_file): # Clean up failed attempt
                    try: os.remove(output_file)
                    except OSError: pass

            # Fallback: Re-encode if copy failed
            quality = self.config["quality"]
            preset_cfg = self._get_quality_preset(quality)
            cmd_reencode = [
                "ffmpeg", "-loglevel", "warning", # Show warnings for re-encode
                "-f", "concat", "-safe", "0", "-i", concat_list_file,
                "-c:v", "libx264", "-preset", preset_cfg["preset"], "-crf", str(preset_cfg["crf"]),
                "-vf", f"format={self.config['pix_fmt']}", # Ensure pixel format
                "-an", output_file, "-y"
            ]
            process_reencode = subprocess.run(cmd_reencode, capture_output=True, text=True, timeout=45)
            if process_reencode.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print("  Hard cut (re-encode) successful.")
                return output_file
            else:
                print(f"  Error: Hard cut (re-encode) also failed. Stderr: {process_reencode.stderr.strip()[-200:]}")
                # Last resort: copy first clip if valid
                if clip1_valid:
                     print("  Using first clip as last resort.");
                     shutil.copy2(clip1, output_file); return output_file
                return None
        except Exception as e: # Catch unexpected errors during file ops etc.
            print(f"  Unexpected error during hard cut creation: {e}")
            return None
        finally:
             if os.path.exists(concat_list_file):
                 try: os.remove(concat_list_file)
                 except OSError: pass

    def _get_quality_preset(self, quality_level):
        """Returns encoding preset, CRF, and default bitrate based on quality level."""
        quality_presets = {
            "low": {"preset": "ultrafast", "crf": 28, "video_bitrate": "4M"},
            "medium": {"preset": "medium", "crf": 23, "video_bitrate": "6M"},
            "high": {"preset": "medium", "crf": 20, "video_bitrate": "8M"},
            "ultra": {"preset": "slow", "crf": 18, "video_bitrate": "10M"}
        }
        # Return specified level or default to 'high' if invalid
        selected = quality_presets.get(quality_level, quality_presets["high"])
        # Override bitrate if set explicitly in config (e.g., for final output)
        selected["video_bitrate"] = self.config.get("video_bitrate", selected["video_bitrate"])
        return selected

    # --- THIS IS THE CORRECTED FUNCTION ---
    def _ensure_shorts_resolution(self, input_file, output_file):
        """Ensure video is in 9:16 aspect ratio (e.g., 1080x1920) for YouTube Shorts, preserving audio."""
        target_res = self.config['resolution']
        target_w, target_h = map(int, target_res.split('x'))

        # Check if input file exists and has audio before proceeding
        has_audio = False
        if os.path.exists(input_file):
            try:
                # Quick ffprobe check for *any* audio stream
                probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", input_file]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, timeout=10)
                if result.stdout.strip(): # If any audio stream index is returned
                    has_audio = True
                else:
                    print(f"Note: Input file '{os.path.basename(input_file)}' for resolution check has no detectable audio stream. Output will be silent.")
            except subprocess.CalledProcessError:
                 # ffprobe returns error if no stream of the selected type exists
                 print(f"Note: Input file '{os.path.basename(input_file)}' for resolution check has no detectable audio stream. Output will be silent.")
                 has_audio = False
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
                 print(f"Warning: Could not probe for audio stream in '{os.path.basename(input_file)}' before resolution check: {e}. Proceeding without audio copy.")
                 has_audio = False # Safer to assume no audio if probe fails
        else:
            print(f"Error: Input file for resolution check missing: {input_file}")
            return None


        cmd = [
            "ffmpeg", "-i", input_file,
            # Scale video to fit target_res, preserve aspect ratio by adding black bars
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-c:v", self.config["video_codec"],
            "-preset", "medium", # Use a reasonable preset for this final step
            "-b:v", self.config["video_bitrate"],
            "-maxrate", self.config["video_bitrate"], # Limit max rate
            "-bufsize", "2M", # Reasonable buffer size
            "-pix_fmt", self.config["pix_fmt"],
            "-r", str(self.config["fps"]),
        ]

        # Conditionally copy audio only if it was detected in the input
        if has_audio:
             cmd.extend(["-c:a", "copy"]) # <-- SOLUTION: Copy existing audio stream
        else:
             cmd.extend(["-an"]) # <-- Explicitly disable audio if none was found

        cmd.extend([output_file, "-y"])

        try:
            print(f"Adjusting resolution to {target_res} for {os.path.basename(input_file)}...")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300) # Reasonable timeout
            if os.path.exists(output_file) and os.path.getsize(output_file) > 100: # Basic check
                 print("Resolution adjustment successful.")
                 return output_file
            else:
                 print(f"Error: Ensuring Shorts resolution produced an invalid file: {output_file}")
                 return None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Error ensuring Shorts resolution: {e}")
            print(f"FFmpeg stderr (Shorts Resolution):\n{err_msg}")
            return None
    # --- END OF CORRECTED FUNCTION ---


    # --- create_beat_synchronized_video ---
    def create_beat_synchronized_video(self):
        """
        Create a video synchronized to music beats, respecting duration limits,
        and applying final audio/video processing including fades.
        """
        if len(self.beat_times) < 2:
            raise ValueError(f"Not enough beat times ({len(self.beat_times)}) to create segments. Need at least 2 (start and end).")
        if not self.video_clips:
            raise ValueError("No valid video clips found to create video from.")

        # --- Initial Duration Handling ---
        # Ensure duration config (if set) respects Shorts limit, adjust effective_max_time and beats if needed
        user_max_duration = self.config.get("duration")
        shorts_limit = 60.0
        if user_max_duration is not None and user_max_duration > shorts_limit:
            print(f"Warning: Configured duration ({user_max_duration}s) exceeds YouTube Shorts limit ({shorts_limit}s). Clamping to {shorts_limit}s.")
            self.config["duration"] = shorts_limit
            # Adjust effective_max_time only if it was previously longer than the new limit
            if self.effective_max_time > shorts_limit:
                 print(f"Re-adjusting effective max time from {self.effective_max_time:.2f}s to {shorts_limit:.2f}s due to clamping.")
                 self.effective_max_time = shorts_limit
                 # Re-filter beat times based on the new effective_max_time
                 print("Re-filtering beat times after duration clamp...")
                 original_beat_count = len(self.beat_times)
                 self.beat_times = self.beat_times[self.beat_times <= self.effective_max_time]
                 # Ensure end time is still present after filtering
                 if len(self.beat_times) == 0 or self.beat_times[-1] < self.effective_max_time - 0.05:
                     if self.effective_max_time > 0: # Avoid adding 0 twice if effective_max_time became 0
                        self.beat_times = np.append(self.beat_times, self.effective_max_time)
                 self.beat_times = np.unique(self.beat_times) # Ensure sorted unique
                 print(f"Beat times adjusted from {original_beat_count} to {len(self.beat_times)}.")
                 if len(self.beat_times) < 2:
                      raise ValueError("Not enough beat times remaining after duration clamp. Cannot create segments.")

        # Config shortcuts
        transition_duration_config = self.config["transition_duration"]
        min_clip_dur_config = self.config["min_clip_duration"]
        max_clip_dur_config = self.config["max_clip_duration"]
        # Min segment duration needs to be slightly > 0, consider transition overlap? Not strictly needed for concat demuxer.
        min_segment_duration_safe = 0.1 # Absolute minimum segment duration after speed changes etc.

        created_segments_metadata = []
        total_processed_duration = 0.0 # Track sum of final segment durations

        print(f"\n--- Creating {len(self.beat_times) - 1} Video Segments (Target End: {self.beat_times[-1]:.2f}s) ---")
        num_segments_to_create = len(self.beat_times) - 1
        previous_clip_path = None

        # --- Segment Creation Loop ---
        for i in range(num_segments_to_create):
            start_time = self.beat_times[i]
            end_time = self.beat_times[i+1]
            target_duration = end_time - start_time

            # Skip segment if interval is too small (e.g., due to duplicate beats somehow)
            if target_duration <= 0.01:
                print(f"  Warning: Skipping segment {i+1} due to near-zero target duration ({target_duration:.3f}s).")
                continue

            print(f"\nSegment {i+1}/{num_segments_to_create} ({start_time:.2f}s -> {end_time:.2f}s)")
            print(f"  Target Duration: {target_duration:.2f}s")

            # Apply min/max segment duration constraints from config
            segment_duration = np.clip(target_duration, min_clip_dur_config, max_clip_dur_config)
            duration_changed = False
            if abs(segment_duration - target_duration) > 0.01: duration_changed = True

            if duration_changed:
                 print(f"  Adjusted Duration (Min/Max): {segment_duration:.2f}s (Original target: {target_duration:.2f}s)")


            # --- Select Clip & Handle Duration/Speed Constraints ---
            clip_info = self._select_clip(previous_clip=previous_clip_path, sequence_position=i, total_sequences=num_segments_to_create)
            clip_path = clip_info["path"]
            speed = clip_info["speed"]
            visual_effect = clip_info["visual_effect"]
            is_reverse = clip_info["reverse"]

            # Calculate required source duration based on target segment duration and speed
            source_duration_needed = segment_duration * speed

            original_clip_duration = self.get_clip_duration(clip_path)
            if original_clip_duration is None:
                 print(f"  Error: Cannot get duration for selected clip {os.path.basename(clip_path)}, skipping segment.")
                 continue # Skip this segment

            # --- Attempt to find a suitable clip (long enough) ---
            max_attempts = 3
            attempt = 0
            clip_found_suitable = False
            while attempt < max_attempts:
                 # Check if the current clip is long enough
                 if original_clip_duration >= source_duration_needed:
                      clip_found_suitable = True
                      break # Found suitable clip

                 attempt += 1
                 print(f"  Warning: Clip {os.path.basename(clip_path)} ({original_clip_duration:.2f}s) too short for {source_duration_needed:.2f}s @ {speed:.2f}x. Re-selecting (Attempt {attempt}/{max_attempts})...")

                 # Try selecting a different clip
                 new_clip_info = self._select_clip(previous_clip=clip_path, # Avoid immediate repeat
                                                    sequence_position=i, total_sequences=num_segments_to_create)

                 # Update clip info and re-evaluate
                 clip_info = new_clip_info
                 clip_path = clip_info["path"]
                 speed = clip_info["speed"] # Re-evaluate speed for the new clip
                 visual_effect = clip_info["visual_effect"] # Re-evaluate effect
                 is_reverse = clip_info["reverse"] # Re-evaluate reverse

                 original_clip_duration = self.get_clip_duration(clip_path)
                 if original_clip_duration is None:
                      print(f"    -> Error getting duration for new clip {os.path.basename(clip_path)}, stopping attempts for this segment.")
                      break # Exit while loop

                 source_duration_needed = segment_duration * speed # Recalculate needed duration
                 print(f"    -> Trying Clip: {os.path.basename(clip_path)} ({original_clip_duration:.2f}s). Need {source_duration_needed:.2f}s @ {speed:.2f}x")


            # --- Handle case where no suitable clip was found ---
            if not clip_found_suitable:
                 if original_clip_duration is None: # Failed to get duration during attempts
                     print(f"  Error: Could not verify duration for clips. Skipping segment {i+1}.")
                     continue

                 # Clip is still too short after attempts, must adjust segment duration to fit this clip
                 print(f"  Warning: No clip found long enough after {max_attempts} attempts. Forcing segment {i+1} to fit {os.path.basename(clip_path)} ({original_clip_duration:.2f}s).")
                 source_duration_needed = original_clip_duration # Use the full available duration of the clip
                 segment_duration = source_duration_needed / speed # Recalculate the final duration of this segment based on the source and speed
                 segment_duration = max(min_segment_duration_safe, segment_duration) # Ensure it's not too short

                 # Check against safe minimum again AFTER recalculation
                 if segment_duration < min_segment_duration_safe:
                      print(f"  Error: Shortened segment duration ({segment_duration:.2f}s) still below safe minimum ({min_segment_duration_safe:.2f}s). Skipping segment.")
                      continue
                 print(f"   -> New Final Segment Duration: {segment_duration:.2f}s")
                 clip_start = 0 # Start from beginning since we are using the whole clip
            else:
                 # Clip is long enough, choose random start point within the available range
                 max_start = original_clip_duration - source_duration_needed
                 clip_start = random.uniform(0, max_start) if max_start > 0 else 0

            # Update previous clip path for next iteration's selection logic
            previous_clip_path = clip_path

            print(f"  Using Clip: {os.path.basename(clip_path)}")
            print(f"    Source: [{clip_start:.2f}s - {clip_start + source_duration_needed:.2f}s] ({source_duration_needed:.2f}s)")
            print(f"    Effects: Speed={speed:.2f}x, Effect='{visual_effect}', Reverse={is_reverse}")
            print(f"    => Producing Segment of ~{segment_duration:.2f}s")


            # --- FFmpeg Processing Chain for the Segment ---
            rand_id = random.randint(1000, 9999)
            base_segment_file = os.path.join(self.temp_dir, f"seg_{i:03d}_{rand_id}_base.mp4")
            rev_segment_file = os.path.join(self.temp_dir, f"seg_{i:03d}_{rand_id}_rev.mp4")
            eff_segment_file = os.path.join(self.temp_dir, f"seg_{i:03d}_{rand_id}_eff.mp4")
            final_segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4") # Standardized final name for concat

            current_input_file = clip_path
            processing_failed = False
            temp_files_created = [] # Keep track of intermediates for this segment

            try:
                # 1. Extract the base segment
                print(f"    1. Extracting...", end=" ", flush=True)
                extract_cmd = [
                    "ffmpeg", "-loglevel", "error", # Less verbose logs for success
                    "-ss", f"{clip_start:.3f}", "-i", current_input_file,
                    "-t", f"{source_duration_needed:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", # Good intermediate quality
                    "-an", "-avoid_negative_ts", "make_zero", # No audio, handle timestamps
                    base_segment_file, "-y"
                ]
                subprocess.run(extract_cmd, check=True, capture_output=True, timeout=60)
                if not (os.path.exists(base_segment_file) and os.path.getsize(base_segment_file) > 100): # Basic check
                    raise ValueError(f"Base extract failed or produced empty file for segment {i+1}")
                current_input_file = base_segment_file; temp_files_created.append(base_segment_file)
                print(f"OK -> {os.path.basename(base_segment_file)}")

                # 2. Reverse (if needed)
                if is_reverse:
                     print(f"    2. Reversing...", end=" ", flush=True)
                     reverse_cmd = [
                         "ffmpeg", "-loglevel", "error", "-i", current_input_file,
                         "-vf", "reverse",
                         "-c:v", "libx264", "-preset", "medium", "-crf", "20", # Consistent intermediate quality
                         "-an", rev_segment_file, "-y"
                     ]
                     subprocess.run(reverse_cmd, check=True, capture_output=True, timeout=60)
                     if not (os.path.exists(rev_segment_file) and os.path.getsize(rev_segment_file) > 100):
                         raise ValueError(f"Reverse failed or produced empty file for segment {i+1}")
                     current_input_file = rev_segment_file; temp_files_created.append(rev_segment_file)
                     print(f"OK -> {os.path.basename(rev_segment_file)}")

                # 3. Visual Effect (if needed and not 'none')
                if visual_effect != "none":
                    print(f"    3. Applying Effect '{visual_effect}'...", end=" ", flush=True)
                    eff_output = self._apply_visual_effect(current_input_file, eff_segment_file, visual_effect)
                    if eff_output is None: raise ValueError(f"Effect '{visual_effect}' failed for segment {i+1}")
                    # Check if effect was actually applied (created new file) vs just copied (effect was 'none' or failed fallback)
                    if eff_output == eff_segment_file and os.path.exists(eff_segment_file):
                         current_input_file = eff_segment_file; temp_files_created.append(eff_segment_file)
                         print(f"OK -> {os.path.basename(eff_segment_file)}")
                    else: # Effect was 'none' or failed fallback copy happened, eff_segment_file might not be the main file
                         current_input_file = eff_output # Use the file path returned (likely the input copied)
                         if eff_output != eff_segment_file and os.path.exists(eff_segment_file):
                             temp_files_created.append(eff_segment_file) # Track the potentially unused output file too for cleanup
                         print(f"OK (Effect Skipped/Copied)")


                # 4. Speed Change (if speed is not 1.0x) -> Produces final_segment_file
                print(f"    4. Applying Speed {speed:.2f}x...", end=" ", flush=True)
                speed_output = self._apply_speed_effect(current_input_file, final_segment_file, speed)
                if speed_output is None: raise ValueError(f"Speed change ({speed:.2f}x) failed for segment {i+1}")
                # speed_output should be final_segment_file on success or if speed was 1.0
                current_input_file = final_segment_file # This is the file we'll use for concatenation
                print(f"OK -> {os.path.basename(final_segment_file)}")


                # --- Final Segment Verification ---
                # Verify the final segment file exists and get its actual duration
                if not (os.path.exists(final_segment_file) and os.path.getsize(final_segment_file) > 100):
                    print(f"  Error: Final segment file {os.path.basename(final_segment_file)} is invalid or empty after processing.")
                    processing_failed = True
                else:
                    final_duration = self.get_clip_duration(final_segment_file) # Use ffprobe for accuracy
                    if final_duration is None or final_duration <= 0.01:
                         print(f"  Error: Could not verify duration or duration too short for final segment {os.path.basename(final_segment_file)}.")
                         processing_failed = True
                    else:
                         print(f"  Segment {i+1} OK (Actual Duration: {final_duration:.2f}s)")
                         segment_info = {
                             "index": i,
                             "file": final_segment_file,
                             "final_duration": final_duration,
                         }
                         created_segments_metadata.append(segment_info)
                         total_processed_duration += final_duration

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
                 # Ensure print starts on a new line if previous print didn't end with one
                 sys.stdout.write('\n')
                 print(f"  Error processing segment {i+1}: {e}")
                 if hasattr(e, 'stderr') and e.stderr:
                     # Decode stderr only if it's bytes
                     err_lines = e.stderr.decode(errors='ignore').strip().splitlines() if isinstance(e.stderr, bytes) else str(e.stderr).strip().splitlines()
                     print(f"  FFmpeg stderr (last 5 lines):\n    "+"\n    ".join(err_lines[-5:]))
                 processing_failed = True
            except Exception as e: # Catch any other unexpected errors
                sys.stdout.write('\n')
                print(f"  Unexpected error processing segment {i+1}: {type(e).__name__} - {e}")
                import traceback
                traceback.print_exc(limit=2, file=sys.stderr) # Print short traceback
                processing_failed = True

            finally:
                # Clean up intermediate files for this segment if it failed, EXCEPT the final segment file if it exists (might be needed for debugging)
                if processing_failed:
                    print(f"  Cleaning up intermediates for failed segment {i+1} (keeping final segment file if it exists)...")
                    # Don't add final_segment_file here, keep it if it was created before failure
                else:
                     # If successful, still remove intermediates other than the final one
                     temp_files_created.append(final_segment_file) # Add final file temporarily to list

                # Remove all files except the final one if processing succeeded, or all if failed AND final doesn't exist/is invalid
                for f in temp_files_created:
                     should_remove = True
                     if not processing_failed and f == final_segment_file:
                         should_remove = False # Keep the successful final segment
                     if processing_failed and f == final_segment_file and os.path.exists(f) and os.path.getsize(f) > 100:
                         should_remove = False # Keep the final file even on failure for debugging, if it seems valid

                     if should_remove and os.path.exists(f):
                          try:
                              # print(f"    Removing temp: {os.path.basename(f)}") # Debug logging
                              os.remove(f)
                          except OSError as remove_e: print(f"    Warning: Could not remove temp file {f}: {remove_e}")
                # End Segment Loop

        print(f"\n--- Finished Segment Creation: {len(created_segments_metadata)} segments generated. ---")
        print(f"--- Total duration of generated segments: {total_processed_duration:.2f}s ---")


        if not created_segments_metadata:
            raise ValueError("No valid video segments were created. Cannot proceed.")


        # --- Concatenate Segments ---
        # Using concat demuxer. This method is simple but sensitive to variations
        # between segments (resolution, fps, timebase). Re-encoding during concat is safer.
        print(f"\n--- Combining {len(created_segments_metadata)} Video Segments using FFmpeg Concat Demuxer ---")
        concat_list_file = os.path.join(self.temp_dir, f"final_concat_list_{self.timestamp}.txt")
        valid_segment_count = 0
        with open(concat_list_file, 'w', encoding='utf-8') as f:
            for segment_meta in created_segments_metadata:
                segment_file = segment_meta['file']
                if os.path.exists(segment_file) and os.path.getsize(segment_file) > 100:
                    # Escape path for ffmpeg concat list
                    safe_path = os.path.abspath(segment_file).replace("'", "'\\''")
                    f.write(f"file '{safe_path}'\n")
                    valid_segment_count += 1
                else:
                    print(f"  Warning: Skipping invalid segment file {segment_file} during concat list creation.")

        if valid_segment_count < 1:
             raise ValueError("No valid segments remain for final concatenation!")
        print(f"Concatenating {valid_segment_count} valid segments.")

        silent_output_video = os.path.join(self.temp_dir, f"silent_output_{self.timestamp}.mp4")
        quality = self.config["quality"]
        preset_cfg = self._get_quality_preset(quality) # Get preset/crf/bitrate based on quality

        # Try re-encoding first for robustness (handles variations in segments better)
        concat_cmd_reencode = [
            "ffmpeg", "-loglevel", "warning", # Quieter logs unless error
            "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-c:v", self.config["video_codec"], # Use configured codec
            "-preset", preset_cfg["preset"],
            "-crf", str(preset_cfg["crf"]),
            "-vf", f"fps=fps={self.config['fps']},format={self.config['pix_fmt']}", # Force FPS and pixel format for consistency
            "-an", silent_output_video, "-y"
        ]
        # Fallback: Try codec copy (faster but less reliable if segments differ)
        concat_cmd_copy = [
            "ffmpeg", "-loglevel", "warning",
            "-f", "concat", "-safe", "0", "-i", concat_list_file,
            "-c", "copy", # Try direct copy
            "-an", silent_output_video, "-y"
        ]

        concat_succeeded = False
        try:
            print("Attempting FFmpeg video concatenation (re-encoding)...")
            subprocess.run(concat_cmd_reencode, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
            if os.path.exists(silent_output_video) and os.path.getsize(silent_output_video) > 100:
                print("Concatenation (re-encoding) successful.")
                concat_succeeded = True
            else:
                print("Concatenation (re-encoding) ran but output is invalid. Will try codec copy.")
                if os.path.exists(silent_output_video): os.remove(silent_output_video) # Clean up failed attempt
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
            print(f"Concatenation (re-encoding) failed: {err_msg[-300:]}. Attempting codec copy...")
            if os.path.exists(silent_output_video): os.remove(silent_output_video)

        # Try copy if re-encode failed or produced invalid output
        if not concat_succeeded:
            try:
                print("Attempting FFmpeg video concatenation (codec copy)...")
                subprocess.run(concat_cmd_copy, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
                if os.path.exists(silent_output_video) and os.path.getsize(silent_output_video) > 100:
                     print("Concatenation (copy codec) successful.")
                     concat_succeeded = True
                else:
                     print("Concatenation (copy codec) also produced invalid output.")
                     raise ValueError("Concatenation failed even with codec copy.")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
                 err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
                 print(f"Error during video concatenation (copy codec): {err_msg[-300:]}")
                 raise ValueError("Failed to concatenate video segments using either method.") from e


        # --- Final Muxing: Add Audio and Optional Video Fade ---
        print("\n--- Adding Audio Track and Final Effects ---")
        # Use effective_max_time for precise duration control of the final output
        final_duration = self.effective_max_time
        print(f"Final target duration: {final_duration:.2f}s")

        # --- Prepare Audio Filters ---
        audio_filter_parts = []
        fade_in_dur = self.config["audio_fade_in"]
        if fade_in_dur > 0:
             actual_fade_in = min(fade_in_dur, final_duration) # Limit fade to available duration
             if actual_fade_in > 0.01:
                  audio_filter_parts.append(f"afade=type=in:start_time=0:duration={actual_fade_in:.3f}")

        fade_out_dur = self.config["audio_fade_out"]
        if fade_out_dur > 0 and final_duration > 0.01:
             actual_fade_out = min(fade_out_dur, final_duration) # Limit fade to available duration
             fade_out_start = max(0, final_duration - actual_fade_out)
             # Ensure fade_out doesn't completely overlap fade_in if duration is very short
             if fade_in_dur > 0 and fade_out_start < fade_in_dur: # Check against original config fade_in_dur
                  fade_out_start = min(fade_in_dur, final_duration) # Start fade out right after fade in finishes (or at end if duration is shorter)
                  actual_fade_out = max(0, final_duration - fade_out_start) # Recalculate actual fade duration

             if actual_fade_out > 0.01:
                  audio_filter_parts.append(f"afade=type=out:start_time={fade_out_start:.3f}:duration={actual_fade_out:.3f}")

        audio_filter_str = ",".join(audio_filter_parts)


        # --- Prepare Video Filters (Fade Out) ---
        video_filter_parts = []
        video_fade_out_config = self.config.get("video_fade_out_duration", 0.0) # Default 0 if missing
        apply_video_fade = video_fade_out_config > 0.01 and final_duration > 0.01

        if apply_video_fade:
            print(f"Applying video fade-out: {video_fade_out_config:.2f}s")
            actual_v_fade_out = min(video_fade_out_config, final_duration) # Limit to available duration
            v_fade_out_start = max(0, final_duration - actual_v_fade_out)
            # Use fade filter with black color
            video_filter_parts.append(f"fade=type=out:start_time={v_fade_out_start:.3f}:duration={actual_v_fade_out:.3f}:color=black")

        video_filter_str = ",".join(video_filter_parts)


        # --- Build Final FFmpeg Command ---
        final_cmd = [
            "ffmpeg", "-loglevel", "warning",
            "-i", silent_output_video,     # Input 0: Concatenated video
            "-i", self.audio_path,         # Input 1: Original audio
            "-map", "0:v:0",               # Map video from input 0
            "-map", "1:a:0?",              # Map audio from input 1, '?' makes it optional (robust if audio file lacks audio)
            "-t", f"{final_duration:.3f}", # Set exact output duration
        ]

        # --- Video Codec/Filter Handling ---
        if apply_video_fade:
            # MUST re-encode video if applying fade filter
            print("NOTE: Re-encoding video during final muxing due to video fade.")
            quality = self.config["quality"]
            preset_cfg = self._get_quality_preset(quality) # Get final quality settings
            final_cmd.extend(["-vf", video_filter_str]) # Apply fade filter
            final_cmd.extend([
                "-c:v", self.config["video_codec"],
                "-preset", preset_cfg["preset"],
                "-b:v", preset_cfg["video_bitrate"], # Use final bitrate from quality preset
                "-maxrate", preset_cfg["video_bitrate"], # Constrain max rate
                "-bufsize", "2M", # Reasonable buffer
                "-pix_fmt", self.config["pix_fmt"],
                "-r", str(self.config["fps"])
            ])
        else:
            # Can attempt to copy video stream if no fade needed
            print("NOTE: Attempting to copy video stream during final muxing (no video fade requested).")
            final_cmd.extend(["-c:v", "copy"])

        # --- Audio Codec/Filter Handling ---
        final_cmd.extend([
            "-c:a", self.config["audio_codec"],
            "-b:a", self.config["audio_bitrate"],
            "-ac", "2",     # Force stereo audio
            "-ar", "48000"  # Force 48kHz sample rate (common standard)
        ])
        if audio_filter_str:
            final_cmd.extend(["-af", audio_filter_str]) # Apply audio fades if specified

        # Output File (Initial location before potential resolution check)
        muxed_output_file = os.path.join(self.temp_dir, f"muxed_output_{self.timestamp}.mp4")
        final_cmd.extend([muxed_output_file, "-y"])

        try:
            print("Executing FFmpeg final muxing command...")
            subprocess.run(final_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)

            if not (os.path.exists(muxed_output_file) and os.path.getsize(muxed_output_file) > 100):
                raise ValueError("Final muxing command ran but produced an invalid or empty file.")

            print("Muxing successful.")

            # --- Ensure Shorts Resolution (Final Step) ---
            # This function now correctly copies audio
            print("Ensuring final output meets Shorts resolution requirements...")
            shorts_optimized_output = os.path.join(os.path.dirname(self.output_file),
                                       f"final_{os.path.basename(self.output_file)}") # Use a distinct temp name

            final_processed_path = self._ensure_shorts_resolution(muxed_output_file, shorts_optimized_output)

            if final_processed_path:
                # Replace the original target output path with the resolution-adjusted one
                print(f"Replacing target output with resolution-adjusted file: {os.path.basename(final_processed_path)}")
                # Ensure target directory exists
                os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
                # Use replace for atomic rename where possible
                os.replace(final_processed_path, self.output_file)
            else:
                # If resolution check failed, use the muxed output directly
                print("\nWarning: Could not optimize for YouTube Shorts resolution. Using the muxed output directly.")
                os.replace(muxed_output_file, self.output_file)


            print(f"\nDone! Output saved as: {self.output_file}")
            final_output_duration = self.get_clip_duration(self.output_file)
            if final_output_duration: print(f"Final verified duration: {final_output_duration:.2f}s")
            else: print("Warning: Could not verify final output video duration.")

            return self.output_file
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
             err_msg = e.stderr.decode(errors='ignore').strip() if hasattr(e, 'stderr') and e.stderr else str(e)
             print(f"Error during final muxing/encoding/resolution check: {e}")
             print(f"FFmpeg stderr (Final Stage):\n{err_msg}")
             # Attempt to clean up intermediate muxed/shorts files if they exist
             if os.path.exists(muxed_output_file): os.remove(muxed_output_file)
             if 'shorts_optimized_output' in locals() and os.path.exists(shorts_optimized_output): os.remove(shorts_optimized_output)
             raise ValueError("Failed final video processing stage.") from e


    # --- create_config_file, load_config_file, cleanup ---
    def create_config_file(self, config_file=None):
        """Save current configuration to a JSON file"""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(self.output_file), f"video_config_{self.timestamp}.json")

        config_to_save = self.config.copy()
        try:
            os.makedirs(os.path.dirname(config_file), exist_ok=True) # Ensure dir exists
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=4, sort_keys=True)
            print(f"Configuration saved to: {config_file}")
            return config_file
        except IOError as e:
            print(f"Error saving configuration file: {e}")
            return None

    def load_config_file(self, config_file):
        """Load configuration from a JSON file, updating existing config"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            print(f"Loading configuration from: {config_file}")
            # Filter loaded keys against defaults? Or allow extra? Allow extra.
            valid_keys = self.default_config.keys()
            loaded_keys = list(loaded_config.keys())
            # Update existing config dictionary
            self.config.update(loaded_config)
            # Report unknown keys (compared to defaults)
            unknown_keys = [k for k in loaded_keys if k not in valid_keys]
            if unknown_keys:
                 print(f"  Note: Loaded keys not in defaults: {unknown_keys}")
            print(f"Configuration updated from file.")
            return self.config
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading configuration file '{config_file}': {e}")
            print("Using previously loaded or default configuration.")
            return None # Indicate failure

    def cleanup(self):
        """Delete temporary files and directory"""
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Temporary directory deleted: {self.temp_dir}")
            except OSError as e:
                print(f"Warning: Error deleting temporary directory {self.temp_dir}: {e}")
        else:
            print("Temporary directory not found, no cleanup needed.")

# --- Main Execution Block ---
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = "/Users/dev/womanareoundtheworld/Music_sync/input"  # Changed to specified music directory
    default_output_dir = os.path.join(script_dir, "output")

    parser = argparse.ArgumentParser(
        description="Creates a rhythmic video synchronized to music beats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Input/Output ---
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--input-dir", "-i", type=str, default=default_input_dir, help="Directory containing video clips.")
    io_group.add_argument("--audio", "-a", type=str, required=False, help="Path to the audio file. If omitted, searches input-dir.")
    io_group.add_argument("--output-dir", "-od", type=str, default=default_output_dir, help="Directory for output video and config.")
    io_group.add_argument("--output-name", "-on", type=str, default=None, help="Output video filename (no extension, timestamp added). Default: rhythmic_video_TIMESTAMP.mp4")
    io_group.add_argument("--temp-dir", type=str, default=None, help="Directory for temporary files. Default: temp_OUTPUTNAME inside output-dir.")
    io_group.add_argument("--keep-temp", "-k", action="store_true", help="Keep temporary files after completion.")
    io_group.add_argument("--config", "-cfg", type=str, default=None, help="Load configuration from JSON file (overrides other args).")
    io_group.add_argument("--duration", "-dur", type=float, default=None, help="Maximum output video duration (seconds). Trims audio/beats accordingly. Default: full audio length up to 60s.")

    # --- Beat Detection ---
    beat_group = parser.add_argument_group('Beat Detection')
    beat_group.add_argument("--beat-sensitivity", type=float, default=1.0, help="Adjusts number of detected beats (1.0=normal, >1 more, <1 less).")
    beat_group.add_argument("--beat-offset", type=float, default=0.0, help="Shift beat timings (seconds).")
    beat_group.add_argument("--beat-grouping", type=int, default=1, help="Use every Nth beat (e.g., 2 for half the beats).")
    beat_group.add_argument("--min-beats", type=int, default=20, help="Minimum beats required within duration; generates fallback if fewer.")
    beat_group.add_argument("--no-musical-structure", action="store_true", help="Use simpler onset strength beat detection instead of combined approach.")

    # --- Clip & Segment Control ---
    clip_group = parser.add_argument_group('Clip & Segment Control')
    clip_group.add_argument("--min-segment", type=float, default=0.5, help="Minimum desired segment length (seconds) before adjustments.")
    clip_group.add_argument("--max-segment", type=float, default=5.0, help="Maximum desired segment length (seconds) before adjustments.")
    clip_group.add_argument("--no-speed-variation", action="store_true", help="Disable random clip speed changes (all clips 1.0x).")
    clip_group.add_argument("--min-speed", type=float, default=0.5, help="Minimum clip speed multiplier (if speed variation enabled).")
    clip_group.add_argument("--max-speed", type=float, default=2.0, help="Maximum clip speed multiplier (if speed variation enabled).")
    clip_group.add_argument("--speed-change-prob", type=float, default=0.3, help="Probability (0-1) of changing clip speed (if enabled).")
    clip_group.add_argument("--reverse-prob", type=float, default=0.15, help="Probability (0-1) of reversing a clip (if long enough).")

    # --- Transitions ---
    trans_group = parser.add_argument_group('Transitions (Note: Transitions beyond hard cuts are experimental with concat demuxer)')
    trans_group.add_argument("--transition", "-t", type=str, default="random", choices=list(EnhancedRhythmicVideoEditor.TRANSITION_TYPES.keys()), help="Default transition type to aim for (visual result may vary).")
    trans_group.add_argument("--transition-duration", "-d", type=float, default=0.3, help="Transition duration (seconds) - primarily affects selection logic.")
    trans_group.add_argument("--hard-cut-ratio", type=float, default=0.2, help="Approximate ratio (0-1) of selecting 'none' (hard cuts) vs. other transitions.")
    trans_group.add_argument("--no-transition-variation", action="store_true", help="Disable random transition type selection (uses --transition type).")
    trans_group.add_argument("--transition-safe", action="store_true", help="Use only basic transitions (fade, crossfade, none) in selection logic.")

    # --- Visuals & Quality ---
    visual_group = parser.add_argument_group('Visuals & Quality')
    visual_group.add_argument("--quality", "-q", type=str, default="high", choices=["low", "medium", "high", "ultra"], help="Output video encoding quality preset (affects CRF/bitrate).")
    visual_group.add_argument("--visual-effects", action="store_true", help="Enable random visual effects on clips.")
    visual_group.add_argument("--effect-prob", type=float, default=0.2, help="Probability (0-1) of applying visual effect per clip (if enabled).")

    # --- Randomization Tuning ---
    rand_group = parser.add_argument_group('Randomization Tuning')
    rand_group.add_argument("--rand-clip-strength", type=float, default=0.8, help="Smart clip selection strength (0=fully random, 1=strictly avoid repeats/prefer unused).")
    rand_group.add_argument("--rand-speed-logic", action="store_true", help="Use fully random speed range regardless of position (if speed variation enabled).")
    rand_group.add_argument("--rand-force-effect", action="store_true", help="Always try to apply random effect (if --visual-effects enabled, overrides --effect-prob).")
    rand_group.add_argument("--rand-transitions", action="store_true", help="Select transitions purely randomly (overrides positional logic).")

    # --- Output Formatting ---
    output_group = parser.add_argument_group('Output Formatting')
    output_group.add_argument("--audio-fade-in", type=float, default=1.0, help="Audio fade-in duration (seconds).")
    output_group.add_argument("--audio-fade-out", type=float, default=3.0, help="Audio fade-out duration (seconds).")
    output_group.add_argument("--video-fadeout-duration", "-vfod", type=float, default=0.0, help="Video fade-to-black duration at the end (seconds). WARNING: Requires re-encoding the final video.") # Existing argument

    args = parser.parse_args()

    # --- Post-process Args & Setup ---
    if not os.path.isdir(args.input_dir): print(f"Error: Input directory not found: {args.input_dir}"); return 1

    # Find audio file
    audio_file = args.audio
    if not audio_file:
        print(f"Audio file not specified, searching in '{args.input_dir}'...")
        audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
        found_audio = []
        for ext in audio_extensions:
            found_audio.extend(glob(os.path.join(args.input_dir, f'*{ext.lower()}')))
            found_audio.extend(glob(os.path.join(args.input_dir, f'*{ext.upper()}')))
        found_audio = sorted(list(set(found_audio)))
        if not found_audio: print(f"Error: No audio file specified and none found in '{args.input_dir}'. Use --audio."); return 1
        if len(found_audio) > 1: print(f"Warning: Multiple audio files found. Using the first one: {os.path.basename(found_audio[0])}")
        audio_file = found_audio[0]
        print(f"Using found audio file: {os.path.basename(audio_file)}")
    elif not os.path.isfile(audio_file): print(f"Error: Specified audio file not found: {audio_file}"); return 1

    # Setup output paths
    try: os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e: print(f"Error creating output directory '{args.output_dir}': {e}"); return 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename_base = args.output_name if args.output_name else f"rhythmic_video_{timestamp}"
    output_file = os.path.join(args.output_dir, f"{output_filename_base}.mp4")
    temp_dir = args.temp_dir if args.temp_dir else os.path.join(args.output_dir, f"temp_{output_filename_base}")


    # --- Create Initial Configuration Dictionary from Args ---
    # Start with defaults, then update with CLI args
    editor_config_init = EnhancedRhythmicVideoEditor(args.input_dir, audio_file).default_config.copy()
    # Only update duration if provided via CLI, otherwise keep default (60s)
    if args.duration is not None:
        editor_config_init["duration"] = args.duration

    # Get quality preset using a temporary editor instance
    temp_editor = EnhancedRhythmicVideoEditor(args.input_dir, audio_file)
    preset_cfg = temp_editor._get_quality_preset(args.quality)
    editor_config_init['video_bitrate'] = preset_cfg['video_bitrate']

    editor_config_init.update({
        # Beat Detection
        "beat_sensitivity": args.beat_sensitivity, "beat_offset": args.beat_offset, "beat_grouping": args.beat_grouping, "min_beats": args.min_beats, "use_musical_structure": not args.no_musical_structure,
        # Clip & Segment
        "min_clip_duration": args.min_segment, "max_clip_duration": args.max_segment, "clip_speed_variation": not args.no_speed_variation, "min_speed": args.min_speed, "max_speed": args.max_speed, "speed_change_probability": args.speed_change_prob, "reverse_clip_probability": args.reverse_prob,
        # Transitions
        "transition_type": args.transition, "transition_duration": args.transition_duration, "hard_cut_ratio": args.hard_cut_ratio, "transition_variation": not args.no_transition_variation, "transition_safe": args.transition_safe,
        # Visuals & Quality
        "quality": args.quality, "apply_visual_effects": args.visual_effects, "visual_effect_probability": args.effect_prob,
        # Randomization
        "randomize_clip_selection_strength": args.rand_clip_strength, "randomize_speed_logic": args.rand_speed_logic, "force_random_effect": args.rand_force_effect, "randomize_transition_selection": args.rand_transitions,
        # Output Formatting
        "audio_fade_in": args.audio_fade_in, "audio_fade_out": args.audio_fade_out,
        "video_fade_out_duration": args.video_fadeout_duration,
    })


    # --- Initialize Editor ---
    try:
        editor = EnhancedRhythmicVideoEditor(
            clips_dir=args.input_dir,
            audio_path=audio_file,
            output_file=output_file,
            temp_dir=temp_dir,
            config=editor_config_init # Pass the fully constructed config
        )
    except (OSError, ImportError) as e:
         print(f"Error during editor initialization: {e}")
         return 1


    # --- Load Config File (Overrides args if specified) ---
    if args.config:
        print("-" * 40)
        if os.path.exists(args.config):
             loaded_conf = editor.load_config_file(args.config)
             if loaded_conf is None: print(f"Warning: Failed to load config file '{args.config}'. Using args/defaults.")
             else: print("Config file loaded and applied, overriding corresponding arguments.")
        else: print(f"Warning: Config file specified but not found: {args.config}. Using args/defaults.")
        print("-" * 40)


    # --- Execute Process ---
    exit_code = 0
    try:
        print("\n" + "=" * 60 + "\n Starting Enhanced Rhythmic Video Generation\n" + "=" * 60)
        # Print key settings *after* potential config load
        print(f" Input Clips Dir : {editor.clips_dir}")
        print(f" Audio Source    : {os.path.basename(editor.audio_path)}")
        print(f" Output Target   : {editor.output_file}")
        print(f" Temporary Dir   : {editor.temp_dir}")
        max_dur_setting = editor.config.get('duration')
        print(f" Max Duration    : {'Full audio (up to 60s)' if max_dur_setting is None else f'{max_dur_setting:.2f}s'}")
        print(f" Video Quality   : {editor.config.get('quality', 'N/A')}")
        print(f" Video Fade Out  : {editor.config.get('video_fade_out_duration', 0.0):.2f}s")
        print("-" * 60)

        # Main steps
        editor.find_video_clips()
        editor.detect_beats()
        final_output_path = editor.create_beat_synchronized_video()

        # Save the *actually used* configuration
        config_save_path = os.path.join(args.output_dir, f"{output_filename_base}_config_used.json")
        editor.create_config_file(config_file=config_save_path)

        print("\n" + "=" * 60 + "\n Process Completed Successfully\n" + "=" * 60)
        print(f" Final video saved to: {final_output_path}")

    except (FileNotFoundError, ValueError, IOError, ImportError) as e:
        print("\n" + "!" * 60 + "\n Configuration or Input Error\n" + "!" * 60, file=sys.stderr)
        print(f" Error: {e}", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        exit_code = 1
    except Exception as e:
        import traceback
        print("\n" + "!" * 60 + "\n Unexpected Error During Processing\n" + "!" * 60, file=sys.stderr)
        print(f" Error Type: {type(e).__name__}", file=sys.stderr)
        print(f" Error Message: {str(e)}", file=sys.stderr)
        print(" Traceback:", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        print("-" * 60, file=sys.stderr)
        exit_code = 1
    finally:
        # Cleanup logic
        if not args.keep_temp and exit_code == 0:
             print("-" * 60); editor.cleanup()
        elif args.keep_temp: print("-" * 60 + f"\nTemporary files kept in: {editor.temp_dir}")
        else: print("-" * 60 + f"\nError occurred, temporary files kept for debugging in: {editor.temp_dir}")

    return exit_code

if __name__ == "__main__":
    # Check for FFmpeg/ffprobe before starting
    ffmpeg_ok = False
    ffprobe_ok = False
    try:
        # Use -version which typically exits 0 if found
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ffmpeg_ok = True
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        ffprobe_ok = True
    except FileNotFoundError:
        if not ffmpeg_ok: print("Error: ffmpeg not found in system PATH. Please install FFmpeg.", file=sys.stderr)
        if not ffprobe_ok: print("Error: ffprobe not found in system PATH. Please install FFmpeg.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # This might happen if -version isn't supported but the command exists
        print(f"Warning: Could not verify ffmpeg/ffprobe versions via '-version' (command failed: {e}). Assuming they exist if no FileNotFoundError occurred.")
        # We don't exit here, just warn. The script will fail later if they truly don't work.
    except Exception as e:
        print(f"An unexpected error occurred while checking for ffmpeg/ffprobe: {e}", file=sys.stderr)
        sys.exit(1)

    if not ffmpeg_ok or not ffprobe_ok:
        print("Aborting due to missing FFmpeg or ffprobe.", file=sys.stderr)
        sys.exit(1)

    # Execute the main function
    sys.exit(main())