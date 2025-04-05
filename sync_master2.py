#!/usr/bin/env python3
import subprocess
import os
import numpy as np
import argparse
import random
from glob import glob
import librosa
from scipy.signal import find_peaks
import shutil
import sys
import json
from collections import defaultdict
from datetime import datetime
import time

class EnhancedRhythmicVideoEditor:
    """
    Enhanced Rhythmic Video Editor that synchronizes video clips to music beats
    with advanced transition effects, speed controls, visual enhancements,
    duration control, and optional intro/outro sequences.
    """

    # Define transition types
    TRANSITION_TYPES = {
        "none": "No transition (hard cut)",
        "crossfade": "Smooth crossfade between clips",
        "fade": "Fade to black/color and back", # Updated description
        "wipe_left": "Wipe from left to right",
        "wipe_right": "Wipe from right to left",
        "zoom_in": "Zoom in transition (radial)", # Clarified type
        "zoom_out": "Zoom out transition (fade)", # Clarified type (uses fade as fallback)
        "random": "Random selection from available transitions"
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
        "random": "Random selection from available effects"
    }

    def __init__(self, clips_dir, audio_path, output_file=None, temp_dir=None, config=None):
        """Initialize the editor with paths and configuration"""
        self.start_time = time.time() # Record start time
        self.clips_dir = os.path.abspath(clips_dir) # Use absolute paths
        self.audio_path = os.path.abspath(audio_path)

        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Determine output directory and filename
        if output_file is None:
            # Default output in the *parent* directory of clips_dir or current dir
            output_parent_dir = os.path.dirname(self.clips_dir) or '.'
            output_file = os.path.join(output_parent_dir, f"rhythmic_video_{self.timestamp}.mp4")
        self.output_file = os.path.abspath(output_file)
        output_dir = os.path.dirname(self.output_file)

        # Determine temp directory
        if temp_dir is None:
            temp_dir = os.path.join(output_dir, f"temp_{self.timestamp}")
        self.temp_dir = os.path.abspath(temp_dir)

        # Default configuration
        self.default_config = {
            # Beat detection parameters
            "beat_sensitivity": 1.0,
            "min_beats": 15, # Adjusted default
            "beat_offset": 0.0,
            "beat_grouping": 1,
            "use_musical_structure": True,

            # Clip selection parameters
            "min_clip_duration": 0.5, # Renamed from min_segment internally
            "max_clip_duration": 5.0, # Renamed from max_segment internally
            "clip_speed_variation": True,
            "min_speed": 0.75, # Adjusted default
            "max_speed": 1.5,  # Adjusted default
            "speed_change_probability": 0.3,
            "reverse_probability": 0.05, # Added config key

            # Transition parameters
            "transition_type": "random", # Default to random
            "transition_duration": 0.3,  # Adjusted default
            "hard_cut_ratio": 0.2,       # Adjusted default
            "transition_variation": True,
            "transition_safe": True,    # Default to safe transitions
            "fallback_transition": "crossfade",

            # Visual parameters
            "resolution": "source", # Kept for potential future use, but dynamically determined now
            "quality": "high",
            "apply_visual_effects": False,
            "visual_effect_probability": 0.15, # Adjusted default

            # Rhythm parameters (less actively used currently, but kept)
            "sync_strength": 0.8,
            "ignore_weak_beats": True,

            # Advanced parameters
            "smart_clip_selection": True,
            "avoid_clip_repetition": True,
            "max_clip_repeats": 3, # Adjusted default
            "scene_variety": 0.7, # Kept for potential future use

            # --- Output Control Parameters ---
            "max_total_duration": 0,      # Max video duration in seconds (0 = no limit)
            "add_intro": False,           # Add a fade-from-color intro
            "add_outro": False,           # Add a fade-to-color outro
            "intro_duration": 2.0,        # Default intro duration
            "outro_duration": 3.0,        # Default outro duration
            "intro_outro_color": "black", # Color for fades (e.g., "black", "white")
            "audio_fade_in": 1.0,         # Audio fade in duration
            "audio_fade_out": 3.0,        # Audio fade out duration
        }

        # Override defaults with provided configuration
        self.config = self.default_config.copy()
        if config:
            # Filter None values from incoming config if necessary
            filtered_config = {k: v for k, v in config.items() if v is not None}
            self.config.update(filtered_config)
            print("Configuration loaded and updated from command line arguments/file.")
            # Optionally print the final effective config:
            # print("Effective configuration:")
            # print(json.dumps(self.config, indent=2))


        # Ensure the temporary directory exists and is empty
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            os.makedirs(self.temp_dir, exist_ok=True)
        except OSError as e:
             raise RuntimeError(f"Could not create or clean temporary directory {self.temp_dir}: {e}")

        # Ensure output directory exists
        try:
             os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        except OSError as e:
             raise RuntimeError(f"Could not create output directory {os.path.dirname(self.output_file)}: {e}")


        # Lists for data
        self.video_clips = []
        self.beat_times = []
        self.segments = [] # Keep track of segment metadata if needed later
        self.clip_durations = {}  # Cache for clip durations
        self.clip_resolutions = {} # Cache for clip resolutions
        self.clip_usage_count = defaultdict(int)  # Track clip usage

        print("\n--- Enhanced Rhythmic Video Editor Initialized ---")
        print(f"Input Clips:   {self.clips_dir}")
        print(f"Audio File:    {self.audio_path}")
        print(f"Output File:   {self.output_file}")
        print(f"Temp Directory:{self.temp_dir}")
        print("-" * 50)


    def _log_time(self, message):
        """Logs a message with elapsed time since initialization."""
        elapsed_time = time.time() - self.start_time
        print(f"[{elapsed_time:7.2f}s] {message}") # Wider padding for time

    def _run_ffmpeg_command(self, cmd, operation_desc="FFmpeg operation", timeout=60):
        """Runs an FFmpeg command with logging and error handling."""
        self._log_time(f"Starting: {operation_desc}")
        # print(f"  Command: {' '.join(cmd)}") # Uncomment for deep FFmpeg debugging
        try:
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True, # Raise CalledProcessError on non-zero exit code
                timeout=timeout
            )
            # Success if check=True doesn't raise error
            self._log_time(f"Finished: {operation_desc} (Success)")
            # Log stderr even on success as it often contains useful info/warnings
            if process.stderr:
                 # print(f"  FFmpeg Info/Warnings:\n{process.stderr.strip()}") # Optional: reduce noise
                 pass
            return True, process.stdout, process.stderr
        except FileNotFoundError:
            print(f"\nFATAL ERROR: 'ffmpeg' command not found.")
            print("Please ensure FFmpeg is installed and accessible in your system's PATH.")
            raise # Re-raise the error to stop execution
        except subprocess.CalledProcessError as e:
            print(f"\n--- ERROR during: {operation_desc} ---")
            print(f"  Command failed with exit code {e.returncode}")
            print(f"  Command: {' '.join(e.cmd)}") # Log the exact command
            print(f"  FFmpeg stderr output:\n{e.stderr.strip()}") # Log the error output
            return False, e.stdout, e.stderr
        except subprocess.TimeoutExpired as e:
            print(f"\n--- ERROR during: {operation_desc} ---")
            print(f"  Command timed out after {timeout} seconds.")
            print(f"  Command: {' '.join(e.cmd)}")
            stderr_output = e.stderr.strip() if e.stderr else "No stderr output before timeout."
            print(f"  FFmpeg output (if any):\n{stderr_output}")
            return False, e.stdout, stderr_output
        except Exception as e: # Catch other potential errors
            print(f"\n--- UNEXPECTED ERROR during: {operation_desc} ---")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Error Type: {type(e).__name__}")
            print(f"  Error: {e}")
            return False, "", str(e)

    def _get_video_metadata(self, clip_path, get_duration=True, get_resolution=True):
        """Gets duration and/or resolution of a video file using ffprobe, with caching."""
        # Return cached values if available
        cached_duration = self.clip_durations.get(clip_path)
        cached_resolution = self.clip_resolutions.get(clip_path)
        if (not get_duration or cached_duration is not None) and \
           (not get_resolution or cached_resolution is not None):
            return cached_duration, cached_resolution

        if not os.path.exists(clip_path):
             self._log_time(f"Warning: Cannot get metadata, file not found: {clip_path}")
             return None, None

        duration = None
        resolution = None
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0"]
        entries = []
        if get_duration: entries.append("format=duration")
        if get_resolution: entries.append("stream=width,height")
        cmd.extend(["-show_entries", ":".join(entries)])
        cmd.extend(["-of", "default=noprint_wrappers=1:nokey=1", clip_path]) # Simple key=value output

        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True, timeout=10)
            output_lines = result.stdout.strip().split('\n')

            if get_duration:
                try:
                    duration = float(output_lines[0]) # Duration should be first if requested
                    self.clip_durations[clip_path] = duration
                except (ValueError, IndexError):
                    self._log_time(f"Warning: Could not parse duration from ffprobe output for {os.path.basename(clip_path)}")
                    self.clip_durations[clip_path] = None # Mark as failed

            if get_resolution:
                try:
                    # Width/height might be on the same line or next line depending on ffprobe version and entries
                    res_line_index = 1 if get_duration and len(output_lines) > 1 else 0
                    if res_line_index < len(output_lines):
                        width = int(output_lines[res_line_index])
                        if res_line_index + 1 < len(output_lines): # Check if height is on next line
                             height = int(output_lines[res_line_index + 1])
                             resolution = (width, height)
                             self.clip_resolutions[clip_path] = resolution
                        else: # Assume width/height were together if only one line left
                             # This part might need adjustment based on exact ffprobe output format
                             self._log_time(f"Warning: Ambiguous resolution output for {os.path.basename(clip_path)}")
                             self.clip_resolutions[clip_path] = None
                    else:
                         self._log_time(f"Warning: Could not find resolution lines in ffprobe output for {os.path.basename(clip_path)}")
                         self.clip_resolutions[clip_path] = None

                except (ValueError, IndexError):
                    self._log_time(f"Warning: Could not parse resolution from ffprobe output for {os.path.basename(clip_path)}")
                    self.clip_resolutions[clip_path] = None # Mark as failed

            return self.clip_durations.get(clip_path), self.clip_resolutions.get(clip_path)

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            self._log_time(f"Warning: ffprobe failed for {os.path.basename(clip_path)}. Error: {e}")
            if get_duration: self.clip_durations[clip_path] = None
            if get_resolution: self.clip_resolutions[clip_path] = None
            return None, None


    def find_video_clips(self, extensions=None):
        """Find all video clips in the specified directory and analyze them."""
        self._log_time(f"Searching for video clips in: {self.clips_dir}")
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.webm'] # Added webm

        self.video_clips = []
        for ext in extensions:
            # Case-insensitive globbing
            self.video_clips.extend(glob(os.path.join(self.clips_dir, f'[!.]*{ext}'), recursive=False)) # Avoid hidden files
            self.video_clips.extend(glob(os.path.join(self.clips_dir, f'[!.]*{ext.upper()}'), recursive=False))

        # Remove duplicates and sort
        self.video_clips = sorted([os.path.abspath(p) for p in set(self.video_clips)])

        if not self.video_clips:
            # Try searching subdirectories if no clips found in top level
            self._log_time("No clips found in top level, searching subdirectories...")
            for ext in extensions:
                 self.video_clips.extend(glob(os.path.join(self.clips_dir, '**', f'[!.]*{ext}'), recursive=True))
                 self.video_clips.extend(glob(os.path.join(self.clips_dir, '**', f'[!.]*{ext.upper()}'), recursive=True))
            self.video_clips = sorted([os.path.abspath(p) for p in set(self.video_clips)])

            if not self.video_clips:
                raise FileNotFoundError(f"No video clips found in directory {self.clips_dir} or subdirectories with extensions {extensions}")

        num_found = len(self.video_clips)
        self._log_time(f"Found {num_found} potential video clips. Analyzing metadata...")

        # Pre-analyze clips to get durations and resolutions
        valid_clips = []
        start_analysis_time = time.time()
        for i, clip_path in enumerate(self.video_clips):
            analyze_progress = f"Analyzing clip {i+1}/{num_found}: {os.path.basename(clip_path)}"
            elapsed = time.time() - start_analysis_time
            if i > 0:
                 avg_time_per_clip = elapsed / i
                 eta = avg_time_per_clip * (num_found - i)
                 analyze_progress += f" (ETA: {eta:.1f}s)"
            print(analyze_progress, end='\r', flush=True)

            duration, resolution = self._get_video_metadata(clip_path, get_duration=True, get_resolution=True)

            if duration is not None and duration > 0.1 and resolution is not None: # Require minimal duration and valid resolution
                valid_clips.append(clip_path)
            else:
                 print(f"\n  Skipping clip: Invalid metadata (Duration: {duration}, Resolution: {resolution}) - {os.path.basename(clip_path)}")

        print("\nClip analysis complete.                                  ") # Clear progress line
        self.video_clips = valid_clips # Update list to only valid ones

        if not self.video_clips:
             raise ValueError("No valid video clips with usable metadata found after analysis.")

        self._log_time(f"Finished finding and analyzing clips. Usable clips: {len(self.video_clips)}")
        # Optionally print first few valid clips
        # print("Usable clips:")
        # for clip in self.video_clips[:5]: print(f" - {os.path.basename(clip)} ({self.clip_durations.get(clip):.2f}s, {self.clip_resolutions.get(clip)})")
        # if len(self.video_clips) > 5: print("   ...")

        return self.video_clips

    def detect_beats(self):
        """Advanced beat detection with adjustable sensitivity and musical structure analysis."""
        self._log_time(f"Analyzing audio file for beats: {os.path.basename(self.audio_path)}...")

        sensitivity = self.config["beat_sensitivity"]
        min_beats_config = self.config["min_beats"]

        # Load audio
        try:
             y, sr = librosa.load(self.audio_path, sr=None) # Load with native sample rate
             duration = librosa.get_duration(y=y, sr=sr)
             if duration <= 0: raise ValueError("Audio file has zero or negative duration.")
             self._log_time(f"Audio loaded. Duration: {duration:.2f}s, Sample Rate: {sr}Hz")
        except Exception as e:
             raise RuntimeError(f"Error loading audio file {self.audio_path}: {e}")


        # Enhanced beat detection using multiple methods
        if self.config["use_musical_structure"]:
            self._log_time("Using musical structure analysis for beat detection...")
            try:
                 hop_length = 512
                 tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=False, units='frames')
                 beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
                 self._log_time(f"Initial beat tracking found {len(beat_times)} beats. Estimated Tempo: {tempo:.2f} BPM")

                 onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
                 onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=True)
                 onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
                 self._log_time(f"Onset detection found {len(onset_times)} onsets.")

                 all_beats = np.unique(np.round(np.concatenate([beat_times, onset_times]), decimals=3))

                 min_beat_interval = 0.15
                 filtered_beats = []
                 if len(all_beats) > 0:
                     filtered_beats.append(all_beats[0])
                     for beat in all_beats[1:]:
                         if beat - filtered_beats[-1] >= min_beat_interval:
                             filtered_beats.append(beat)
                 beat_times = np.array(filtered_beats)
                 self._log_time(f"Combined and filtered beats: {len(beat_times)} remaining.")

                 if sensitivity != 1.0 and len(beat_times) > 2 : # Need at least 3 beats to remove some
                     self._log_time(f"Applying beat sensitivity factor: {sensitivity:.2f}")
                     if sensitivity < 1.0:
                         removal_prob = 1.0 - sensitivity
                         temp_beats = [beat_times[0]] # Keep first
                         for i in range(1, len(beat_times) - 1): # Iterate over middle beats
                             if random.random() > removal_prob:
                                 temp_beats.append(beat_times[i])
                         temp_beats.append(beat_times[-1]) # Keep last
                         beat_times = np.array(temp_beats)
                         self._log_time(f"Beats after sensitivity adjustment: {len(beat_times)}")
                     # else: Sensitivity > 1.0 currently doesn't add beats in this implementation

            except Exception as e:
                 print(f"\nWarning: Advanced beat detection failed: {e}. Falling back to simpler method.")
                 self.config["use_musical_structure"] = False # Fallback flag
                 beat_times = None # Reset beat_times

        # Simpler beat detection (or fallback)
        if not self.config["use_musical_structure"] or beat_times is None:
             self._log_time("Using simpler onset strength peak picking for beat detection...")
             hop_length = 512
             onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
             # Adjust threshold based on sensitivity and onset envelope properties
             threshold = np.median(onset_env) + np.std(onset_env) * (sensitivity - 1.0) # Adjust std based on sensitivity
             threshold = max(np.min(onset_env) * 1.1, threshold) # Ensure threshold is slightly above minimum

             min_distance_sec = 0.18 # Slightly shorter min distance
             min_distance_frames = int(librosa.time_to_frames(min_distance_sec, sr=sr, hop_length=hop_length))
             peaks, _ = find_peaks(onset_env, height=threshold, distance=min_distance_frames)
             beat_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
             self._log_time(f"Simple beat detection found {len(beat_times)} beats.")

        # Apply beat grouping
        if self.config["beat_grouping"] > 1 and len(beat_times) > 0:
            group_size = self.config["beat_grouping"]
            self._log_time(f"Grouping beats into groups of {group_size}...")
            grouped_beats = beat_times[::group_size]
            # Ensure the last beat is included if grouping caused it to be skipped
            if len(beat_times) > 0 and beat_times[-1] > grouped_beats[-1] + 1e-3: # Check if last beat is significantly after last grouped beat
                 grouped_beats = np.append(grouped_beats, beat_times[-1])
            beat_times = np.unique(np.round(grouped_beats, decimals=3)) # Ensure uniqueness after potential append
            self._log_time(f"Beats after grouping: {len(beat_times)}")

        # Apply beat offset
        if self.config["beat_offset"] != 0.0:
             offset = self.config["beat_offset"]
             self._log_time(f"Applying beat offset of {offset:.3f} seconds...")
             beat_times = beat_times + offset
             beat_times = beat_times[(beat_times >= 0) & (beat_times <= duration)] # Clip to valid range
             self._log_time(f"Beats after offset: {len(beat_times)}")


        # Ensure minimum number of beats
        if len(beat_times) < min_beats_config:
            self._log_time(f"Warning: Only {len(beat_times)} beats detected (minimum set to {min_beats_config}).")
            if duration > 0:
                 print(f"Generating artificial beats to meet the minimum...")
                 num_beats_to_add = min_beats_config - len(beat_times)
                 # Generate evenly spaced beats within the duration
                 # Avoid placing beats too close to existing ones if possible (more complex)
                 # Simple approach: linspace over the duration
                 artificial_beats = np.linspace(0, duration, min_beats_config + 2)[1:-1] # Generate target number between 0 and duration
                 # Combine and ensure uniqueness
                 beat_times = np.unique(np.round(np.concatenate((beat_times, artificial_beats)), decimals=3))
                 # Trim if we overshoot due to combining with existing beats
                 if len(beat_times) > min_beats_config * 1.5: # Heuristic trim if too many
                      beat_times = np.linspace(0, duration, min_beats_config + 2)[1:-1]
                 self._log_time(f"Combined/Generated beats: {len(beat_times)}")
            else:
                 self._log_time("Cannot generate artificial beats for zero duration audio.")


        # Ensure beats near start and end
        start_threshold = 0.5
        end_threshold = 0.5

        final_beats = list(beat_times)
        if len(final_beats) == 0 or final_beats[0] > start_threshold:
             final_beats.insert(0, 0.0)
             self._log_time("Added beat at time 0.0s for starting segment.")

        if len(final_beats) == 0 or final_beats[-1] < duration - end_threshold:
             final_beats.append(duration)
             self._log_time(f"Added beat at time {duration:.2f}s for ending segment.")

        # Final check for duplicates and sort
        self.beat_times = np.unique(np.round(np.array(final_beats), decimals=3))
        # Ensure beats are monotonically increasing
        self.beat_times = np.sort(self.beat_times)

        # Remove beats that are too close after all adjustments
        min_final_interval = 0.1 # Absolute minimum interval
        valid_beat_indices = np.where(np.diff(self.beat_times, prepend=self.beat_times[0]-1) >= min_final_interval)[0]
        self.beat_times = self.beat_times[valid_beat_indices]


        self._log_time(f"Final beat detection complete. Total usable beats: {len(self.beat_times)}")

        if len(self.beat_times) > 20:
            print(f"First 10 beat times: {self.beat_times[:10]}")
            print(f"Last 10 beat times: {self.beat_times[-10:]}")
        else:
            print(f"Beat times: {self.beat_times}")

        if len(self.beat_times) < 2:
             raise ValueError("Beat detection resulted in less than 2 usable beats. Cannot proceed.")

        return self.beat_times

    def _select_clip(self, previous_clip=None, sequence_position=0, total_sequences=1):
        """Intelligently select a clip based on configuration settings."""
        # Already filtered to valid clips in find_video_clips
        available_clips = self.video_clips.copy()

        if not available_clips:
             print("Warning: No valid clips available for selection.")
             return None

        # Apply smart clip selection if enabled
        if self.config["smart_clip_selection"]:
            available_clips.sort(key=lambda clip: self.clip_usage_count[clip])
            max_repeats = self.config.get("max_clip_repeats", 0) # Default 0 means unlimited
            if self.config["avoid_clip_repetition"] and max_repeats > 0:
                filtered_available = [clip for clip in available_clips if self.clip_usage_count[clip] < max_repeats]
                if filtered_available:
                     available_clips = filtered_available
                else:
                    # All clips hit max repeats, allow selection from least used among them
                    print("Note: All clips reached max usage, selecting least used.")
                    available_clips.sort(key=lambda clip: self.clip_usage_count[clip])


        # Avoid repeating the immediate previous clip if possible
        if previous_clip is not None and previous_clip in available_clips and len(available_clips) > 1:
            available_clips.remove(previous_clip)

        if not available_clips:
             # Fallback: If filtering removed all options (e.g., only one clip total and it was previous)
             available_clips = self.video_clips.copy() # Go back to full list
             available_clips.sort(key=lambda clip: self.clip_usage_count[clip]) # Sort again
             if previous_clip is not None and previous_clip in available_clips and len(available_clips) > 1:
                 available_clips.remove(previous_clip) # Try removing previous again
             if not available_clips: # If still none (only one clip total)
                  available_clips = self.video_clips # Use the single clip


        # Select a clip (use the first one from the sorted list - least used)
        clip_path = available_clips[0]
        self.clip_usage_count[clip_path] += 1

        # Determine speed variation
        speed = 1.0
        if self.config["clip_speed_variation"] and random.random() < self.config["speed_change_probability"]:
            min_speed = self.config["min_speed"]
            max_speed = self.config["max_speed"]
            # Bias towards center, allow full range
            speed = random.uniform(min_speed, max_speed)

        # Determine visual effect
        visual_effect = "none"
        if self.config["apply_visual_effects"] and random.random() < self.config["visual_effect_probability"]:
            effects_pool = list(self.VISUAL_EFFECTS.keys())
            effects_pool = [e for e in effects_pool if e not in ["random", "none"]]
            if effects_pool:
                 visual_effect = random.choice(effects_pool)

        # Determine reverse effect
        is_reverse = random.random() < self.config.get("reverse_probability", 0.05)

        return {
            "path": clip_path,
            "speed": speed,
            "visual_effect": visual_effect,
            "reverse": is_reverse
        }

    def _select_transition(self, segment_index, total_segments):
        """Select a transition type based on configuration."""
        # Handle hard cut ratio first
        if random.random() < self.config["hard_cut_ratio"]:
            return "none"

        requested_type = self.config["transition_type"]
        use_variation = self.config["transition_variation"]
        use_safe_mode = self.config.get("transition_safe", True)
        fallback_type = self.config.get("fallback_transition", "crossfade")
        safe_transitions = ["none", "crossfade", "fade"]

        # Determine the pool of possible transitions
        possible_transitions = list(self.TRANSITION_TYPES.keys())
        possible_transitions = [t for t in possible_transitions if t not in ["random", "none"]] # Exclude control types

        if use_safe_mode:
             possible_transitions = [t for t in possible_transitions if t in safe_transitions]
             if not possible_transitions: return fallback_type # Fallback if safe pool is empty

        if not use_variation:
             # Use requested type if valid within current mode (safe/unsafe)
             if requested_type in possible_transitions:
                 return requested_type
             else:
                 # Requested type not allowed (e.g., wipe in safe mode), use fallback
                 print(f"Note: Requested transition '{requested_type}' not allowed in current mode. Using fallback '{fallback_type}'.")
                 return fallback_type


        # --- Variation is ON ---
        # Select randomly from the determined pool (safe or full)
        if not possible_transitions: # Should have been caught earlier, but safeguard
             return fallback_type
        # Could add weighting here based on segment_index if desired
        return random.choice(possible_transitions)


    def _apply_visual_effect(self, input_file, output_file, effect):
        """Apply a visual effect to a clip."""
        # Define FFmpeg filters for each effect
        effect_filters = {
            "none": None, # Indicate no filter needed
            "grayscale": "hue=s=0",
            "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
            "vibrance": "eq=saturation=1.5:contrast=1.1",
            "vignette": "vignette=angle=PI/4:aspect=1",
            "blur_edges": "vignette=angle=PI/5,gblur=sigma=5",
            "sharpen": "unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount=1.0",
            "mirror": "hflip",
            "flip": "vflip"
        }
        filter_str = effect_filters.get(effect)

        if filter_str:
            cmd = [
                "ffmpeg", "-i", input_file,
                "-vf", filter_str,
                "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                "-an", output_file, "-y"
            ]
            success, _, _ = self._run_ffmpeg_command(cmd, f"Applying {effect} effect", timeout=45)
            if success: return output_file
            else:
                 print(f"  Warning: Failed applying {effect}. Using input segment.")
                 try: shutil.copy2(input_file, output_file); return output_file
                 except Exception as e: print(f"  Error copying fallback: {e}"); return None
        else: # No effect needed, just copy if names differ
             if input_file != output_file:
                  try: shutil.copy2(input_file, output_file); return output_file
                  except Exception as e: print(f"  Error copying for 'none' effect: {e}"); return None
             return input_file # Input is already the output

    def _apply_speed_effect(self, input_file, output_file, speed):
        """Apply speed adjustment to a clip."""
        if abs(speed - 1.0) < 0.01: # Check for effective no-change
            if input_file != output_file:
                try: shutil.copy2(input_file, output_file); return output_file
                except Exception as e: print(f"  Error copying for 1x speed: {e}"); return None
            return input_file # Input is already the output

        pts_filter = f"setpts={1.0/speed:.4f}*PTS"
        cmd = [
            "ffmpeg", "-i", input_file,
            "-vf", pts_filter,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-an", output_file, "-y"
        ]
        success, _, _ = self._run_ffmpeg_command(cmd, f"Applying {speed:.2f}x speed", timeout=45)
        if success: return output_file
        else:
            print(f"  Warning: Failed applying speed {speed:.2f}x. Using input segment.")
            try: shutil.copy2(input_file, output_file); return output_file
            except Exception as e: print(f"  Error copying fallback: {e}"); return None

    def _apply_reverse_effect(self, input_file, output_file):
        """Reverses a video clip."""
        cmd = [
            "ffmpeg", "-i", input_file,
            "-vf", "reverse",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-an", output_file, "-y"
        ]
        success, _, _ = self._run_ffmpeg_command(cmd, f"Reversing segment", timeout=45)
        if success: return output_file
        else:
            print(f"  Warning: Failed reversing segment. Using input segment.")
            try: shutil.copy2(input_file, output_file); return output_file
            except Exception as e: print(f"  Error copying fallback: {e}"); return None


    def _create_transition(self, clip1_path, clip2_path, output_file, transition_type, duration):
        """Creates a transition between two clips using FFmpeg's xfade filter."""
        self._log_time(f"Creating '{transition_type}' transition ({duration:.2f}s)")

        # Basic validation
        if not os.path.exists(clip1_path) or os.path.getsize(clip1_path) < 100:
             print(f"  Error: Clip 1 invalid for transition: {os.path.basename(clip1_path)}")
             return self._copy_fallback(clip2_path, output_file, "clip 2") # Use clip 2 if clip 1 fails
        if not os.path.exists(clip2_path) or os.path.getsize(clip2_path) < 100:
             print(f"  Error: Clip 2 invalid for transition: {os.path.basename(clip2_path)}")
             return self._copy_fallback(clip1_path, output_file, "clip 1") # Use clip 1 if clip 2 fails


        # Get durations to potentially clamp transition duration
        clip1_dur, _ = self._get_video_metadata(clip1_path, get_duration=True, get_resolution=False)
        clip2_dur, _ = self._get_video_metadata(clip2_path, get_duration=True, get_resolution=False)

        effective_duration = max(0.1, duration) # Ensure min duration
        if clip1_dur is not None and clip2_dur is not None:
             max_possible = min(clip1_dur, clip2_dur)
             if effective_duration > max_possible:
                  print(f"  Note: Clamping transition duration from {effective_duration:.2f}s to {max_possible*0.9:.2f}s (90% of shortest clip)")
                  effective_duration = max(0.1, max_possible * 0.9)


        # xfade filter names might differ slightly or need specific parameters
        transition_filters = {
            "crossfade": f"xfade=transition=fade:duration={effective_duration:.3f}:offset=0",
            "fade": f"xfade=transition=fadeblack:duration={effective_duration:.3f}:offset=0", # Fade thru black
            "wipe_left": f"xfade=transition=wipeleft:duration={effective_duration:.3f}:offset=0",
            "wipe_right": f"xfade=transition=wiperight:duration={effective_duration:.3f}:offset=0",
            "zoom_in": f"xfade=transition=radial:duration={effective_duration:.3f}:offset=0", # Using radial as proxy
            "zoom_out": f"xfade=transition=fade:duration={effective_duration:.3f}:offset=0", # Fallback for zoom out
        }
        filter_str = transition_filters.get(transition_type)

        if not filter_str: # Should not happen if _select_transition works, but safeguard
             print(f"  Error: Invalid transition type '{transition_type}' provided to _create_transition. Using hard cut.")
             return self._create_hard_cut(clip1_path, clip2_path, output_file)

        # Determine resolution from first clip for filtergraph sizing if needed (usually automatic)
        _, clip1_res = self._get_video_metadata(clip1_path, get_duration=False, get_resolution=True)
        # res_str = f"width={clip1_res[0]}:height={clip1_res[1]}" if clip1_res else "" # Potentially add size to filter

        cmd = [
            "ffmpeg",
            "-i", clip1_path, "-i", clip2_path,
            "-filter_complex", f"[0][1]{filter_str}[v]", # Basic xfade
            "-map", "[v]",
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-an", output_file, "-y"
        ]

        success, _, stderr = self._run_ffmpeg_command(cmd, f"Applying {transition_type} transition", timeout=90)

        if success and os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            return output_file
        else:
            print(f"  Error applying '{transition_type}'. Trying fallback '{self.config['fallback_transition']}'.")
            fallback_type = self.config['fallback_transition']
            if fallback_type != "none" and fallback_type != transition_type:
                 fallback_filter_str = transition_filters.get(fallback_type)
                 if fallback_filter_str:
                      fallback_cmd = [
                          "ffmpeg", "-i", clip1_path, "-i", clip2_path,
                          "-filter_complex", f"[0][1]{fallback_filter_str}[v]", "-map", "[v]",
                          "-c:v", "libx264", "-preset", "fast", "-crf", "24", # Faster fallback
                          "-an", output_file, "-y"
                      ]
                      success_fallback, _, _ = self._run_ffmpeg_command(fallback_cmd, f"Fallback {fallback_type} transition", timeout=60)
                      if success_fallback and os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                           print("  Fallback transition successful.")
                           return output_file

            # If fallback fails or is 'none', resort to hard cut
            print("  Fallback failed or not applicable. Performing hard cut.")
            return self._create_hard_cut(clip1_path, clip2_path, output_file)


    def _create_hard_cut(self, clip1_path, clip2_path, output_file):
        """Concatenates two clips using the concat demuxer (more reliable than protocol)."""
        list_filename = f"concat_list_{random.randint(1000, 9999)}.txt"
        concat_list_path = os.path.join(self.temp_dir, list_filename)

        # Check existence right before writing list
        clip1_exists = os.path.exists(clip1_path) and os.path.getsize(clip1_path) > 100
        clip2_exists = os.path.exists(clip2_path) and os.path.getsize(clip2_path) > 100

        if not clip1_exists and not clip2_exists:
             print(f"  Error: Neither clip exists for hard cut. Cannot create combined segment.")
             return None

        try:
             with open(concat_list_path, 'w') as f:
                 if clip1_exists: f.write(f"file '{clip1_path.replace("'", "'\\''")}'\n")
                 if clip2_exists: f.write(f"file '{clip2_path.replace("'", "'\\''")}'\n")

             cmd = [
                 "ffmpeg", "-f", "concat", "-safe", "0",
                 "-i", concat_list_path,
                 "-c", "copy", # Fast stream copy
                 output_file, "-y"
             ]
             success, _, _ = self._run_ffmpeg_command(cmd, f"Creating hard cut (concat)", timeout=45)

             os.remove(concat_list_path) # Clean up list file

             if success and os.path.exists(output_file) and os.path.getsize(output_file) > 100:
                 return output_file
             else:
                 print(f"  Hard cut concatenation command failed. Trying single clip fallback.")
                 if clip1_exists: return self._copy_fallback(clip1_path, output_file, "clip 1")
                 elif clip2_exists: return self._copy_fallback(clip2_path, output_file, "clip 2")
                 else: return None # Should be unreachable

        except Exception as e:
             print(f"  Unexpected error during hard cut: {e}")
             if os.path.exists(concat_list_path): os.remove(concat_list_path)
             # Try single clip fallback on error
             if clip1_exists: return self._copy_fallback(clip1_path, output_file, "clip 1")
             elif clip2_exists: return self._copy_fallback(clip2_path, output_file, "clip 2")
             return None

    def _copy_fallback(self, source_path, dest_path, description):
        """Copies a source file to destination as a fallback action."""
        print(f"  Using fallback: copying {description} to {os.path.basename(dest_path)}")
        try:
             shutil.copy2(source_path, dest_path)
             return dest_path
        except Exception as e:
             print(f"  Error copying fallback file {description}: {e}")
             return None


    def _get_target_resolution_str(self):
        """Determines the target resolution string, probing clips if necessary."""
        target_width, target_height = None, None
        # Try to get resolution from the first *valid* source clip
        first_clip_path = next((clip for clip in self.video_clips), None)
        if first_clip_path:
            self._log_time(f"Probing resolution from first valid clip: {os.path.basename(first_clip_path)}")
            _, resolution = self._get_video_metadata(first_clip_path, get_duration=False, get_resolution=True)
            if resolution:
                target_width, target_height = resolution

        if not target_width or not target_height:
            target_width, target_height = 1280, 720 # Default fallback
            self._log_time(f"Warning: Could not determine source resolution. Using default {target_width}x{target_height}.")
        else:
            self._log_time(f"Determined target resolution: {target_width}x{target_height}")

        return f"{target_width}x{target_height}"

    def _create_fade_segment(self, duration, fade_type='in', color='black', resolution='1280x720', fps=30):
        """Creates a video segment fading in or out from a solid color."""
        if duration <= 0: return None # Cannot create zero duration segment

        base_filename = f"{fade_type}_fade_{color}_{duration:.1f}s_{resolution}_{self.timestamp}.mp4"
        output_file = os.path.join(self.temp_dir, base_filename)
        num_frames = max(1, int(duration * fps)) # Ensure at least 1 frame

        fade_filter = None
        if fade_type == 'in':
            fade_filter = f"fade=type=in:start_frame=0:nb_frames={num_frames}:color={color}"
        elif fade_type == 'out':
            fade_filter = f"fade=type=out:start_frame=0:nb_frames={num_frames}:color={color}"

        cmd = [
            "ffmpeg", "-f", "lavfi",
            "-i", f"color=c={color}:s={resolution}:r={fps}:d={duration}",
        ]
        if fade_filter: cmd.extend(["-vf", fade_filter])
        cmd.extend([
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-tune", "stillimage",
            "-an", output_file, "-y"
        ])

        success, _, _ = self._run_ffmpeg_command(cmd, f"Creating {fade_type} fade segment ({duration:.1f}s)")

        if success and os.path.exists(output_file) and os.path.getsize(output_file) > 100:
            return output_file
        else:
            self._log_time(f"Error: Failed to create {fade_type} fade segment.")
            return None

    def create_beat_synchronized_video(self):
        """Main workflow to create the synchronized video."""
        if len(self.beat_times) < 2: raise ValueError("Not enough beats detected!")
        if not self.video_clips: raise ValueError("No valid video clips found!")

        # Config shortcuts
        min_segment_dur_config = self.config["min_clip_duration"]
        max_segment_dur_config = self.config["max_clip_duration"]
        min_allowed_segment_dur = 0.1 # Absolute minimum

        # --- Determine Target Resolution Early ---
        target_resolution_str = None
        if self.config['add_intro'] or self.config['add_outro']:
             target_resolution_str = self._get_target_resolution_str()


        # --- Generate Individual Segments ---
        self._log_time("--- Stage 1: Generating Beat-Matched Segments ---")
        processed_segments_info = []
        temp_segment_files_to_clean = [] # Track intermediates for this stage
        previous_clip_path = None
        total_beat_intervals = len(self.beat_times) - 1

        for i in range(total_beat_intervals):
            segment_start_time = self.beat_times[i]
            segment_end_time = self.beat_times[i+1]
            target_duration = round(segment_end_time - segment_start_time, 3)

            segment_log_prefix = f"Segment {i+1}/{total_beat_intervals} ({segment_start_time:.2f}s -> {segment_end_time:.2f}s, Target Dur: {target_duration:.2f}s):"
            print(f"\n{segment_log_prefix}")

            if target_duration < min_allowed_segment_dur:
                print(f"  Skipping: Segment duration ({target_duration:.2f}s) below absolute minimum ({min_allowed_segment_dur:.2f}s).")
                continue
            if target_duration < min_segment_dur_config:
                 print(f"  Note: Segment duration ({target_duration:.2f}s) below configured minimum ({min_segment_dur_config:.2f}s). Proceeding.")

            # Clamp duration based on max config
            if target_duration > max_segment_dur_config:
                 print(f"  Clamping segment duration from {target_duration:.2f}s to max configured ({max_segment_dur_config:.2f}s).")
                 target_duration = max_segment_dur_config

            clip_info = self._select_clip(previous_clip_path, i, total_beat_intervals)
            if clip_info is None: print("  Error: Could not select clip. Skipping segment."); continue

            clip_path = clip_info["path"]
            speed = clip_info["speed"]
            visual_effect = clip_info["visual_effect"]
            is_reverse = clip_info["reverse"]
            previous_clip_path = clip_path

            print(f"  Selected Clip: {os.path.basename(clip_path)} (Used {self.clip_usage_count[clip_path]} times)")

            source_duration_needed = target_duration * speed
            clip_actual_duration, _ = self._get_video_metadata(clip_path, get_duration=True, get_resolution=False)

            if clip_actual_duration is None: print(f"  Error: Duration unknown for {os.path.basename(clip_path)}. Skipping."); continue

            clip_start = 0
            if clip_actual_duration <= source_duration_needed:
                source_duration_needed = clip_actual_duration
                print(f"  Note: Clip shorter ({clip_actual_duration:.2f}s) than needed source ({source_duration_needed:.2f}s). Using full clip.")
            else:
                max_start = clip_actual_duration - source_duration_needed
                clip_start = random.uniform(0, max_start)
            print(f"  Using clip from {clip_start:.2f}s for {source_duration_needed:.2f}s (pre-speed).")

            # Define intermediate filenames
            base_name = f"segment_{i:04d}"
            raw_file = os.path.join(self.temp_dir, f"{base_name}_0_raw.mp4")
            rev_file = os.path.join(self.temp_dir, f"{base_name}_1_rev.mp4")
            eff_file = os.path.join(self.temp_dir, f"{base_name}_2_eff.mp4")
            spd_file = os.path.join(self.temp_dir, f"{base_name}_3_spd.mp4") # Final stage output for this segment
            current_intermediates = [raw_file, rev_file, eff_file] # Spd file becomes the final output of this stage

            # --- Process Segment Pipeline ---
            current_input_file = None
            try:
                extract_cmd = [
                    "ffmpeg", "-i", clip_path, "-ss", f"{clip_start:.3f}", "-t", f"{source_duration_needed:.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-an", raw_file, "-y" ]
                success, _, _ = self._run_ffmpeg_command(extract_cmd, f"Extracting raw segment {i+1}")
                if not success: raise Exception("Raw extraction failed.")
                current_input_file = raw_file

                if is_reverse:
                    print(f"  Applying reverse effect.")
                    reversed_output = self._apply_reverse_effect(current_input_file, rev_file)
                    if reversed_output is None: raise Exception("Reverse failed.")
                    current_input_file = reversed_output
                # else: pass through

                if visual_effect != "none":
                    print(f"  Applying visual effect: {visual_effect}")
                    effect_output = self._apply_visual_effect(current_input_file, eff_file, visual_effect)
                    if effect_output is None: raise Exception("Effect failed.")
                    current_input_file = effect_output
                # else: pass through

                if abs(speed - 1.0) > 0.01:
                    print(f"  Applying speed effect: {speed:.2f}x")
                    speed_output = self._apply_speed_effect(current_input_file, spd_file, speed)
                    if speed_output is None: raise Exception("Speed failed.")
                    current_input_file = speed_output # This is the final output for this stage
                else: # If speed is 1.0, copy the last result to the final name
                     shutil.copy2(current_input_file, spd_file)
                     current_input_file = spd_file

                # --- Segment Finalization ---
                final_segment_file = current_input_file # spd_file holds the final result now

                if final_segment_file and os.path.exists(final_segment_file) and os.path.getsize(final_segment_file) > 100:
                     final_duration, _ = self._get_video_metadata(final_segment_file, get_duration=True, get_resolution=False)
                     if final_duration is None: final_duration = target_duration # Estimate if probe fails
                     duration_diff = abs(final_duration - target_duration)
                     if duration_diff > 0.15: # Slightly larger tolerance after effects/speed
                          print(f"  Warning: Final segment duration ({final_duration:.2f}s) differs significantly from target ({target_duration:.2f}s).")

                     next_transition = self._select_transition(i, total_beat_intervals)
                     print(f"  Segment {i+1} OK. Final File: {os.path.basename(final_segment_file)}. Next transition: {next_transition}")

                     segment_info = { "index": i, "file": final_segment_file, "next_transition": next_transition }
                     processed_segments_info.append(segment_info)
                     temp_segment_files_to_clean.extend(current_intermediates) # Add intermediates for cleanup
                else:
                     raise Exception("Final segment file missing or empty after processing pipeline.")

            except Exception as e:
                 print(f"  ERROR processing segment {i+1}: {e}. Skipping segment.")
                 # Attempt cleanup of this segment's intermediates if error occurred
                 for f in current_intermediates + [spd_file]:
                      if os.path.exists(f): try: os.remove(f); except OSError: pass
                 continue # Skip to next segment

        # --- Cleanup Intermediate Segment Files ---
        self._log_time("Cleaning up intermediate segment processing files...")
        cleaned_count = 0
        for f in temp_segment_files_to_clean:
             if os.path.exists(f):
                  try: os.remove(f); cleaned_count += 1
                  except OSError as e: print(f"  Warn: Could not delete temp file {f}: {e}")
        self._log_time(f"Cleaned {cleaned_count} intermediate files.")

        if not processed_segments_info:
             raise ValueError("No segments were successfully created! Check logs.")
        self._log_time(f"--- Stage 1 Complete: {len(processed_segments_info)} Segments Ready ---")

        # --- Stage 2: Apply Transitions and Combine Main Content ---
        self._log_time("--- Stage 2: Applying Transitions & Combining Main Content ---")
        main_content_files = [] # Files to be included in the main concat (post-transition)
        transition_files_to_clean = [] # Track generated transition files

        apply_transitions = self.config["transition_duration"] > 0 and \
                            any(s['next_transition'] != 'none' for s in processed_segments_info[:-1])

        if apply_transitions:
            self._log_time("Applying transitions between main segments...")
            last_processed_segment_index = -1
            num_segments = len(processed_segments_info)

            for i in range(num_segments - 1):
                current_info = processed_segments_info[i]
                next_info = processed_segments_info[i+1]
                transition_type = current_info["next_transition"]

                segment_log_prefix = f"Transition {i+1}/{num_segments-1} ({transition_type}):"
                print(f"\n{segment_log_prefix}")

                if i < last_processed_segment_index:
                     print("  Skipping (already included in previous transition).")
                     continue # This segment was the second part of the previous transition

                if transition_type == "none":
                     print(f"  Hard cut. Adding segment {i+1} ({os.path.basename(current_info['file'])}).")
                     main_content_files.append(current_info['file'])
                     last_processed_segment_index = i
                     continue

                # Create Transition
                trans_output_file = os.path.join(self.temp_dir, f"transition_{i:04d}_{transition_type}.mp4")
                transitioned_file = self._create_transition(
                    current_info['file'], next_info['file'], trans_output_file,
                    transition_type, self.config["transition_duration"]
                )

                if transitioned_file:
                    print(f"  Transition successful -> {os.path.basename(transitioned_file)}")
                    main_content_files.append(transitioned_file)
                    transition_files_to_clean.append(transitioned_file)
                    last_processed_segment_index = i + 1 # Mark that segments i and i+1 are covered
                else:
                    # Transition failed, use fallback (hard cut of current segment)
                    print(f"  Transition failed. Fallback: Adding segment {i+1} directly.")
                    main_content_files.append(current_info['file'])
                    last_processed_segment_index = i

            # Add the very last segment if it wasn't part of the last transition attempt
            if last_processed_segment_index < num_segments - 1:
                 print(f"\nAdding final segment {num_segments} ({os.path.basename(processed_segments_info[-1]['file'])}) to main content list.")
                 main_content_files.append(processed_segments_info[-1]['file'])

        else:
            self._log_time("No transitions applicable. Using segments directly for main content.")
            main_content_files = [s['file'] for s in processed_segments_info]

        if not main_content_files:
             raise ValueError("Main content list is empty after transition processing.")
        self._log_time(f"--- Stage 2 Complete: {len(main_content_files)} Main Content Parts Ready ---")


        # --- Stage 3: Assemble Final Video (Intro -> Main -> Outro) ---
        self._log_time("--- Stage 3: Assembling Final Video ---")
        final_assembly_list = [] # List of absolute paths for final concat
        generated_special_segments = [] # Track intro/outro files

        # 1. Add Intro
        if self.config['add_intro'] and self.config['intro_duration'] > 0 and target_resolution_str:
            self._log_time(f"Creating intro ({self.config['intro_duration']}s fade from {self.config['intro_outro_color']})")
            intro_file = self._create_fade_segment(
                self.config['intro_duration'], 'in', self.config['intro_outro_color'], target_resolution_str)
            if intro_file: final_assembly_list.append(intro_file); generated_special_segments.append(intro_file)

        # 2. Add Main Content
        final_assembly_list.extend(main_content_files)

        # 3. Add Outro
        if self.config['add_outro'] and self.config['outro_duration'] > 0 and target_resolution_str:
            self._log_time(f"Creating outro ({self.config['outro_duration']}s fade to {self.config['intro_outro_color']})")
            outro_file = self._create_fade_segment(
                self.config['outro_duration'], 'out', self.config['intro_outro_color'], target_resolution_str)
            if outro_file: final_assembly_list.append(outro_file); generated_special_segments.append(outro_file)


        # --- Create Final Concatenation List File ---
        concat_list_filename = f"final_assembly_list_{self.timestamp}.txt"
        concat_list_path = os.path.join(self.temp_dir, concat_list_filename)
        self._log_time(f"Creating final assembly list: {concat_list_filename} with {len(final_assembly_list)} entries.")
        valid_assembly_count = 0
        try:
            with open(concat_list_path, 'w', encoding='utf-8') as f:
                for video_file in final_assembly_list:
                    abs_path = os.path.abspath(video_file)
                    if os.path.exists(abs_path) and os.path.getsize(abs_path) > 100:
                         # Escape single quotes for ffmpeg list file format
                         safe_path = abs_path.replace("'", "'\\''")
                         f.write(f"file '{safe_path}'\n")
                         valid_assembly_count += 1
                    else:
                         print(f"  Warning: File missing or empty, skipping from final assembly list: {abs_path}")
        except IOError as e:
             raise IOError(f"Error writing final assembly list file: {e}")

        if valid_assembly_count < 1:
             raise ValueError("Final assembly list contains no valid video files!")


        # --- Concatenate Video Parts (Initial Silent Version) ---
        self._log_time(f"Combining {valid_assembly_count} video parts...")
        silent_output_initial = os.path.join(self.temp_dir, f"silent_output_initial_{self.timestamp}.mp4")

        # Quality presets
        quality_presets = {"low": ("veryfast", "28"), "medium": ("medium", "23"), "high": ("slow", "20"), "ultra": ("slower", "18")}
        preset, crf = quality_presets.get(self.config["quality"], quality_presets["high"])
        self._log_time(f"Using quality preset: {self.config['quality']} (preset={preset}, crf={crf})")

        concat_cmd = [
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
            "-c:v", "libx264", "-preset", preset, "-crf", crf,
            "-an", silent_output_initial, "-y"
        ]
        success_concat, _, _ = self._run_ffmpeg_command(concat_cmd, "Concatenating final video parts (re-encode)", timeout=600) # Longer timeout

        # Fallback concat with stream copy if re-encode failed
        if not success_concat or not os.path.exists(silent_output_initial) or os.path.getsize(silent_output_initial) < 100:
             print("\nWarning: Concatenation with re-encoding failed. Attempting fallback with stream copy...")
             alt_cmd = [
                 "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
                 "-c", "copy", "-an", silent_output_initial, "-y"
             ]
             success_alt, _, _ = self._run_ffmpeg_command(alt_cmd, "Fallback concatenation (stream copy)", timeout=300)
             if not success_alt or not os.path.exists(silent_output_initial) or os.path.getsize(silent_output_initial) < 100:
                  # Cleanup generated special files before raising error
                  for f in generated_special_segments + transition_files_to_clean:
                       if os.path.exists(f): try: os.remove(f); except OSError: pass
                  raise RuntimeError("FATAL: Both concatenation methods failed. Cannot create video.")

        self._log_time(f"Initial combined silent video created: {os.path.basename(silent_output_initial)}")


        # --- Apply Maximum Duration Limit ---
        silent_output_final = silent_output_initial # Assume no truncation initially
        max_dur = self.config.get('max_total_duration', 0)

        if max_dur > 0:
            self._log_time(f"Checking duration limit ({max_dur}s)...")
            initial_duration, _ = self._get_video_metadata(silent_output_initial, get_duration=True, get_resolution=False)

            if initial_duration is None:
                 self._log_time("Warning: Could not get duration. Cannot apply limit.")
            elif initial_duration > max_dur:
                 self._log_time(f"Truncating video from {initial_duration:.2f}s to {max_dur:.2f}s.")
                 silent_output_truncated = os.path.join(self.temp_dir, f"silent_output_truncated_{max_dur}s_{self.timestamp}.mp4")
                 truncate_cmd = [
                     "ffmpeg", "-i", silent_output_initial,
                     "-to", f"{max_dur:.3f}", # Truncate TO timestamp
                     "-c", "copy", "-an", silent_output_truncated, "-y"
                 ]
                 success_truncate, _, _ = self._run_ffmpeg_command(truncate_cmd, f"Truncating video", timeout=180)

                 if success_truncate and os.path.exists(silent_output_truncated):
                      silent_output_final = silent_output_truncated
                      # Remove the non-truncated version
                      if silent_output_initial != silent_output_final and os.path.exists(silent_output_initial):
                           try: os.remove(silent_output_initial); except OSError: pass
                 else:
                      self._log_time(f"Warning: Failed to truncate video. Using full length.")
                      silent_output_final = silent_output_initial
            else:
                 self._log_time(f"Duration ({initial_duration:.2f}s) within limit. No truncation needed.")
                 silent_output_final = silent_output_initial

        # --- Stage 4: Add Audio Track and Finalize ---
        self._log_time(f"--- Stage 4: Adding Audio and Finalizing ---")
        self._log_time(f"Using video base: {os.path.basename(silent_output_final)}")

        # Get durations for fade calculations
        final_video_duration, _ = self._get_video_metadata(silent_output_final, get_duration=True, get_resolution=False)
        if final_video_duration is None:
            self._log_time("Warning: Could not determine final video duration for audio fades. Fades might be inaccurate.")
            # Use a reasonable estimate if needed, e.g., max_dur if set, or initial duration
            final_video_duration = max_dur if max_dur > 0 else 180.0 # Default estimate

        # Prepare audio filters
        audio_filter_parts = []
        fade_in_dur = self.config.get("audio_fade_in", 0.0)
        fade_out_dur = self.config.get("audio_fade_out", 0.0)

        if fade_in_dur > 0:
            audio_filter_parts.append(f"afade=t=in:st=0:d={fade_in_dur:.3f}")
            self._log_time(f"Applying audio fade-in: {fade_in_dur:.1f}s")

        if fade_out_dur > 0 and final_video_duration > 0:
            fade_start = max(0, final_video_duration - fade_out_dur) # Ensure fade start isn't negative
            actual_fade_out_dur = min(fade_out_dur, final_video_duration - fade_start) # Adjust duration if video is short
            if actual_fade_out_dur > 0.1: # Only apply if meaningful duration
                 audio_filter_parts.append(f"afade=t=out:st={fade_start:.3f}:d={actual_fade_out_dur:.3f}")
                 self._log_time(f"Applying audio fade-out: {actual_fade_out_dur:.1f}s (starting at {fade_start:.1f}s)")
            else:
                 self._log_time("Skipping audio fade-out (video too short or fade duration too small).")

        # Final muxing command
        final_cmd = [
            "ffmpeg",
            "-i", silent_output_final,      # Final silent video base
            "-i", self.audio_path,        # Original audio
            "-map", "0:v:0", "-map", "1:a:0?", # Map streams
            "-c:v", "copy",               # Copy processed video stream
            "-c:a", "aac", "-b:a", "192k", # Encode audio
            "-shortest",                  # End when shortest stream (video after truncation) ends
        ]
        if audio_filter_parts: final_cmd.extend(["-af", ",".join(audio_filter_parts)])
        final_cmd.extend([self.output_file, "-y"])

        # Execute final command
        success_final, _, _ = self._run_ffmpeg_command(final_cmd, "Adding audio track and finalizing video", timeout=300)

        if not success_final or not os.path.exists(self.output_file) or os.path.getsize(self.output_file) < 100:
             # Cleanup generated special files before raising error
             for f in generated_special_segments + transition_files_to_clean:
                  if os.path.exists(f): try: os.remove(f); except OSError: pass
             raise RuntimeError("FATAL: Failed to add audio track or finalize the video.")


        # --- Final Duration Check ---
        final_video_duration_str = "N/A"
        final_video_seconds = 0
        try:
            final_video_seconds, _ = self._get_video_metadata(self.output_file, get_duration=True, get_resolution=False)
            if final_video_seconds is not None:
                minutes, seconds = divmod(int(final_video_seconds), 60)
                final_video_duration_str = f"{minutes:02d}:{seconds:02d} ({final_video_seconds:.2f}s)"
        except Exception as e:
            print(f"Warning: Could not determine final video duration after creation: {e}")

        # --- Cleanup Special Segments (Intro/Outro/Transitions) ---
        self._log_time("Cleaning up generated intro/outro/transition segments...")
        cleaned_special_count = 0
        for f in generated_special_segments + transition_files_to_clean:
             if os.path.exists(f):
                  try: os.remove(f); cleaned_special_count += 1
                  except OSError as e: print(f"  Warn: Could not delete temp file {f}: {e}")
        self._log_time(f"Cleaned {cleaned_special_count} special segments.")

        total_elapsed_time = time.time() - self.start_time
        self._log_time(f"--- All Stages Complete ---")
        print(f"\nOutput Video: {self.output_file}")
        print(f"Final Duration: {final_video_duration_str}")
        print(f"Total Time:   {total_elapsed_time:.2f} seconds")

        return self.output_file


    def create_config_file(self, config_file=None):
        """Save current configuration to a JSON file."""
        if config_file is None:
            # Save alongside the output video with matching base name
            base_name = os.path.splitext(self.output_file)[0]
            config_file = f"{base_name}_config.json"

        try:
             # Use the effective self.config which includes defaults, CLI args, and loaded file overrides
             with open(config_file, 'w') as f:
                 json.dump(self.config, f, indent=4, sort_keys=True)
             self._log_time(f"Configuration saved to: {config_file}")
             return config_file
        except IOError as e:
             print(f"Error saving configuration file '{config_file}': {e}")
             return None


    def load_config_file(self, config_file):
        """Load configuration from a JSON file, updating defaults/CLI args."""
        if not os.path.exists(config_file):
             print(f"Warning: Configuration file not found: {config_file}")
             return self.config # Return current config

        try:
             with open(config_file, 'r') as f:
                 loaded_config = json.load(f)

             # Update current config (which already has defaults + CLI args)
             # Loaded file takes precedence
             self.config.update(loaded_config)
             self._log_time(f"Configuration loaded and updated from: {config_file}")
             # print("Effective configuration after loading file:")
             # print(json.dumps(self.config, indent=2))
             return self.config
        except (IOError, json.JSONDecodeError) as e:
             print(f"Error loading or parsing config file {config_file}: {e}")
             return self.config # Return potentially partially updated config


    def cleanup(self):
        """Delete the main temporary directory."""
        if os.path.exists(self.temp_dir):
            try:
                 shutil.rmtree(self.temp_dir)
                 self._log_time(f"Temporary directory deleted: {self.temp_dir}")
            except OSError as e:
                 print(f"Error deleting temporary directory {self.temp_dir}: {e}")
                 print("Please remove it manually if needed.")
        else:
             self._log_time("Temporary directory not found, no cleanup needed.")


# =============================================================================
# Main Execution Block
# =============================================================================
def main():
    overall_start_time = time.time()

    parser = argparse.ArgumentParser(
        description="Creates a video synchronized to music rhythm using clips from an input directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Basic I/O Arguments ---
    io_group = parser.add_argument_group('Input/Output')
    io_group.add_argument("--input", "-i", type=str, required=True, # Make input dir mandatory
                        help="Directory containing input video clips.")
    io_group.add_argument("--audio", "-a", type=str, required=True, # Make audio file mandatory
                        help="Path to the input audio file.")
    io_group.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path. If None, generated automatically based on input dir/timestamp.")
    io_group.add_argument("--output-dir", default=None,
                        help="Directory to save output video and config file (defaults to parent of input dir or current dir).")
    io_group.add_argument("--temp-dir", default=None,
                        help="Directory for temporary files. If None, created automatically.")
    io_group.add_argument("--keep-temp", "-k", action="store_true",
                        help="Keep temporary files after completion for debugging.")
    io_group.add_argument("--config", "-c", type=str,
                        help="Load configuration from a JSON file (overrides defaults and command-line args).")
    io_group.add_argument("--save-config", action="store_true",
                        help="Save the final configuration used to a JSON file alongside the output video.")


    # --- Beat Detection Arguments ---
    beat_group = parser.add_argument_group('Beat Detection')
    beat_group.add_argument("--beat-sensitivity", type=float, default=1.0,
                        help="Beat detection sensitivity (lower=fewer, higher=more).")
    beat_group.add_argument("--min-beats", type=int, default=15,
                            help="Minimum beats required; artificial beats generated if below this.")
    beat_group.add_argument("--beat-offset", type=float, default=0.0,
                        help="Offset detected beat times by +/- seconds.")
    beat_group.add_argument("--beat-grouping", type=int, default=1,
                        help="Use every Nth beat for cuts (e.g., 2 = every other).")
    beat_group.add_argument("--no-musical-structure", action="store_true",
                             help="Disable advanced beat detection (use simpler onset peaks).")

    # --- Clip Selection & Timing Arguments ---
    clip_group = parser.add_argument_group('Clip Selection & Timing')
    clip_group.add_argument("--min-segment", type=float, default=0.5,
                        help="Minimum desired duration for a video segment between beats.")
    clip_group.add_argument("--max-segment", type=float, default=5.0,
                        help="Maximum allowed duration for a video segment (longer beat intervals clamped).")
    clip_group.add_argument("--no-speed-variation", action="store_true",
                        help="Disable random clip speed variation.")
    clip_group.add_argument("--min-speed", type=float, default=0.75,
                        help="Minimum clip speed multiplier.")
    clip_group.add_argument("--max-speed", type=float, default=1.5,
                        help="Maximum clip speed multiplier.")
    clip_group.add_argument("--speed-change-prob", type=float, default=0.3,
                             help="Probability (0-1) of applying speed variation.")
    clip_group.add_argument("--max-clip-repeats", type=int, default=3,
                            help="Max times a clip can be used (0 = unlimited).")
    clip_group.add_argument("--no-smart-select", action="store_true",
                             help="Disable smart clip selection (prioritizes less-used clips).")
    clip_group.add_argument("--reverse-prob", type=float, default=0.05,
                            help="Probability (0-1) of reversing a clip segment.")


    # --- Transition Arguments ---
    trans_group = parser.add_argument_group('Transitions')
    trans_group.add_argument("--transition", "-t", type=str, default="random",
                        choices=list(EnhancedRhythmicVideoEditor.TRANSITION_TYPES.keys()),
                        help="Default transition type or basis for random selection.")
    trans_group.add_argument("--transition-duration", "-d", type=float, default=0.3,
                        help="Transition duration in seconds.")
    trans_group.add_argument("--hard-cut-ratio", type=float, default=0.2,
                        help="Approximate ratio (0-1) of hard cuts vs. transitions.")
    trans_group.add_argument("--no-transition-variation", action="store_true",
                        help="Disable random variation in transition types.")
    # Changed dest and default for clarity: default is SAFE, flag turns it OFF
    trans_group.add_argument("--allow-unsafe-transitions", dest='transition_safe', action='store_false', default=True,
                        help="Allow potentially less compatible transitions (wipes, zooms - requires testing).")
    trans_group.add_argument("--fallback-transition", type=str, default="crossfade",
                        choices=["none", "crossfade", "fade"],
                        help="Fallback transition type if complex one fails.")


    # --- Visual Arguments ---
    visual_group = parser.add_argument_group('Visual Effects & Quality')
    visual_group.add_argument("--quality", "-q", type=str, default="high",
                        choices=["low", "medium", "high", "ultra"], help="Output video quality preset.")
    visual_group.add_argument("--visual-effects", action="store_true",
                        help="Enable random visual effects on some clips.")
    visual_group.add_argument("--visual-effect-prob", type=float, default=0.15,
                             help="Probability (0-1) of applying a random visual effect.")

    # --- Output Control Arguments ---
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument("--max-duration", type=float, default=0,
                        help="Maximum final video duration in seconds (0=no limit). Truncates if longer.")
    output_group.add_argument("--add-intro", action="store_true",
                        help="Add a fade-from-color intro sequence.")
    output_group.add_argument("--intro-duration", type=float, default=2.0,
                        help="Duration of the intro sequence.")
    output_group.add_argument("--add-outro", action="store_true",
                        help="Add a fade-to-color outro sequence.")
    output_group.add_argument("--outro-duration", type=float, default=3.0,
                        help="Duration of the outro sequence.")
    output_group.add_argument("--intro-outro-color", type=str, default="black",
                              help="Color for intro/outro fades (e.g., 'black', 'white', '#FF0000').")
    output_group.add_argument("--audio-fade-in", type=float, default=1.0,
                             help="Audio fade-in duration (seconds).")
    output_group.add_argument("--audio-fade-out", type=float, default=3.0,
                             help="Audio fade-out duration (seconds).")


    # --- Parse Arguments ---
    args = parser.parse_args()

    # --- Prepare Paths ---
    input_dir = os.path.abspath(args.input)
    audio_file = os.path.abspath(args.audio)

    # Determine output directory if not specified
    if args.output_dir is None:
        output_dir = os.path.dirname(input_dir) or '.' # Parent of input or current dir
    else:
        output_dir = os.path.abspath(args.output_dir)

    # Determine output filename if not specified
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output is None:
        output_filename = f"rhythmic_video_{timestamp}.mp4"
        output_file = os.path.join(output_dir, output_filename)
    else:
        # If output path is just a name, join with output_dir
        if os.path.dirname(args.output) == '':
             output_file = os.path.join(output_dir, args.output)
        else:
             output_file = os.path.abspath(args.output)
        # Ensure output directory exists for specified path
        output_dir = os.path.dirname(output_file)

    # Determine temp directory
    if args.temp_dir is None:
        temp_dir = os.path.join(output_dir, f"temp_{timestamp}")
    else:
        temp_dir = os.path.abspath(args.temp_dir)

    # Basic validation
    if not os.path.isdir(input_dir):
        print(f"Error: Input path is not a valid directory: {input_dir}")
        return 1
    if not os.path.isfile(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        return 1


    # --- Create Configuration Dictionary from Args ---
    # We pass this to the Editor, letting its defaults fill in the gaps.
    # We only include args that are *not* None or flags that are True.
    cli_config = {}

    def add_if_set(cfg_key, arg_val):
        if arg_val is not None: cli_config[cfg_key] = arg_val
    def add_if_true(cfg_key, arg_val):
        if arg_val: cli_config[cfg_key] = True # Only set if flag is present
    def add_if_false(cfg_key, arg_val):
        if not arg_val: cli_config[cfg_key] = False # Only set if flag is NOT present (for inverted logic)


    # Map args to config keys (matching default_config keys in __init__)
    add_if_set("beat_sensitivity", args.beat_sensitivity)
    add_if_set("min_beats", args.min_beats)
    add_if_set("beat_offset", args.beat_offset)
    add_if_set("beat_grouping", args.beat_grouping)
    add_if_false("use_musical_structure", not args.no_musical_structure) # Note inversion

    add_if_set("min_clip_duration", args.min_segment) # Map CLI name to internal name
    add_if_set("max_clip_duration", args.max_segment)
    add_if_false("clip_speed_variation", not args.no_speed_variation)
    add_if_set("min_speed", args.min_speed)
    add_if_set("max_speed", args.max_speed)
    add_if_set("speed_change_probability", args.speed_change_prob)
    add_if_set("max_clip_repeats", args.max_clip_repeats)
    add_if_false("smart_clip_selection", not args.no_smart_select)
    add_if_set("reverse_probability", args.reverse_prob)

    add_if_set("transition_type", args.transition)
    add_if_set("transition_duration", args.transition_duration)
    add_if_set("hard_cut_ratio", args.hard_cut_ratio)
    add_if_false("transition_variation", not args.no_transition_variation)
    add_if_set("transition_safe", args.transition_safe) # Directly use parsed value (default True)
    add_if_set("fallback_transition", args.fallback_transition)

    add_if_set("quality", args.quality)
    add_if_true("apply_visual_effects", args.visual_effects)
    add_if_set("visual_effect_probability", args.visual_effect_prob)

    add_if_set("max_total_duration", args.max_duration)
    add_if_true("add_intro", args.add_intro)
    add_if_set("intro_duration", args.intro_duration)
    add_if_true("add_outro", args.add_outro)
    add_if_set("outro_duration", args.outro_duration)
    add_if_set("intro_outro_color", args.intro_outro_color)
    add_if_set("audio_fade_in", args.audio_fade_in)
    add_if_set("audio_fade_out", args.audio_fade_out)


    # --- Initialize and Run Editor ---
    editor = None
    try:
        editor = EnhancedRhythmicVideoEditor(
            clips_dir=input_dir,
            audio_path=audio_file,
            output_file=output_file,
            temp_dir=temp_dir,
            config=cli_config # Pass config derived from CLI args
        )

        # Load config file *after* CLI args processed, potentially overriding them
        if args.config:
             if os.path.exists(args.config):
                 print(f"\nLoading configuration from specified file: {args.config}")
                 editor.load_config_file(args.config)
             else:
                 print(f"\nWarning: Specified config file not found: {args.config}. Using defaults and CLI args.")

        # Run the main process
        editor.find_video_clips()
        editor.detect_beats()
        final_output_file = editor.create_beat_synchronized_video()

        # Save configuration if requested
        if args.save_config:
            editor.create_config_file() # Saves alongside output video

        print("\n--- Video Generation Successful ---")

    except (FileNotFoundError, ValueError, RuntimeError, Exception) as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        # Optional detailed traceback for debugging
        # import traceback
        # print("\nTraceback:")
        # print(traceback.format_exc())
        return 1 # Indicate failure

    finally:
        # --- Cleanup ---
        if editor and not args.keep_temp:
            editor.cleanup()
        elif editor and args.keep_temp:
            print(f"\nTemporary files kept at: {editor.temp_dir}")

        overall_end_time = time.time()
        total_time = overall_end_time - overall_start_time
        print(f"\nTotal Script Execution Time: {total_time:.2f} seconds")

    return 0 # Indicate success

if __name__ == "__main__":
    # Ensure FFmpeg/FFprobe are accessible before starting
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("FATAL ERROR: FFmpeg and/or FFprobe not found or not executable.")
        print("Please install FFmpeg (which includes FFprobe) and ensure it's in your system's PATH.")
        sys.exit(1)

    sys.exit(main())