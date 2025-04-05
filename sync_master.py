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

class EnhancedRhythmicVideoEditor:
    """
    Enhanced Rhythmic Video Editor that synchronizes video clips to music beats
    with advanced transition effects, speed controls, and visual enhancements.
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
        self.clips_dir = clips_dir
        self.audio_path = audio_path
        
        # Generate timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_file is None:
            # Include timestamp in default filename
            output_file = os.path.join(os.path.dirname(clips_dir), f"rhythmic_video_{self.timestamp}.mp4")
        self.output_file = output_file
        
        if temp_dir is None:
            temp_dir = os.path.join(os.path.dirname(output_file), f"temp_{self.timestamp}")
        self.temp_dir = temp_dir
        
        # Default configuration
        self.default_config = {
            # Beat detection parameters
            "beat_sensitivity": 1.0,
            "min_beats": 20,
            "beat_offset": 0.0,  # Offset beats by this many seconds
            "beat_grouping": 1,  # Group every N beats together
            
            # Clip selection parameters
            "min_clip_duration": 0.5,
            "max_clip_duration": 5.0,
            "clip_speed_variation": True,  # Allow speed variations
            "min_speed": 0.5,  # Slowest clip speed (0.5 = half speed)
            "max_speed": 2.0,  # Fastest clip speed (2.0 = double speed)
            "speed_change_probability": 0.3,  # Probability of changing clip speed
            
            # Transition parameters
            "transition_type": "crossfade",  # Default transition type
            "transition_duration": 0.5,  # Transition duration in seconds
            "hard_cut_ratio": 0.3,  # Ratio of hard cuts to transitions
            "transition_variation": True,  # Use varied transition types
            "transition_safe": True,  # Use only the most compatible transitions
            "fallback_transition": "crossfade",  # Fallback transition type
            
            # Visual parameters
            "resolution": "source",  # Output resolution (source, 720p, 1080p, 4K)
            "quality": "high",  # Output quality (low, medium, high, ultra)
            "apply_visual_effects": False,  # Apply visual effects to clips
            "visual_effect_probability": 0.2,  # Probability of applying a visual effect
            
            # Rhythm parameters
            "sync_strength": 0.8,  # How strongly to sync to beats (0.0-1.0)
            "ignore_weak_beats": True,  # Ignore beats below threshold
            "use_musical_structure": True,  # Use musical structure for cuts
            
            # Advanced parameters
            "smart_clip_selection": True,  # Use smart algorithm for clip selection
            "avoid_clip_repetition": True,  # Avoid repeating the same clip consecutively
            "max_clip_repeats": 2,  # Maximum times a clip can repeat in the video
            "scene_variety": 0.7,  # Variety of scenes/clips (0.0-1.0)
            
            # Output parameters
            "add_intro": False,  # Add an intro sequence
            "add_outro": False,  # Add an outro sequence
            "intro_duration": 3.0,  # Intro duration in seconds
            "outro_duration": 3.0,  # Outro duration in seconds
            "audio_fade_in": 1.0,  # Audio fade in duration in seconds
            "audio_fade_out": 3.0,  # Audio fade out duration in seconds
        }
        
        # Override defaults with provided configuration
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Ensure the temporary directory exists and is empty
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Lists for data
        self.video_clips = []
        self.beat_times = []
        self.segments = []
        self.clip_durations = {}  # Cache for clip durations
        self.clip_usage_count = defaultdict(int)  # Track clip usage
        
        # Speed variations for clips (will be populated during processing)
        self.clip_speeds = {}

    def find_video_clips(self, extensions=None):
        """Find all video clips in the specified directory"""
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.wmv']
            
        self.video_clips = []
        for ext in extensions:
            self.video_clips.extend(glob(os.path.join(self.clips_dir, f'*{ext}')))
            
        if not self.video_clips:
            raise FileNotFoundError(f"No video clips found in directory {self.clips_dir}")
            
        print(f"Found {len(self.video_clips)} video clips:")
        for clip in self.video_clips:
            print(f" - {os.path.basename(clip)}")
            
        # Pre-analyze clips to get durations and other metadata
        self._analyze_clips()
            
        return self.video_clips
    
    def _analyze_clips(self):
        """Analyze all clips to gather metadata (duration, resolution, etc.)"""
        print("\nAnalyzing video clips...")
        
        for i, clip_path in enumerate(self.video_clips):
            print(f"Analyzing clip {i+1}/{len(self.video_clips)}: {os.path.basename(clip_path)}")
            
            # Get duration
            duration = self.get_clip_duration(clip_path)
            self.clip_durations[clip_path] = duration
            
            # Get resolution and other metadata if needed in the future
            # This could be expanded to analyze brightness, movement, etc.
        
        print("Clip analysis complete.")
    
    def get_clip_duration(self, clip_path):
        """Get the duration of a clip, with caching"""
        if clip_path in self.clip_durations:
            return self.clip_durations[clip_path]
            
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", clip_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            duration = float(result.stdout.strip())
            self.clip_durations[clip_path] = duration
            return duration
        except ValueError:
            print(f"Warning: Could not determine duration for {clip_path}")
            # Return a reasonable default duration
            return 5.0
    
    def detect_beats(self):
        """
        Advanced beat detection with adjustable sensitivity and musical structure analysis
        """
        print(f"Analyzing audio file: {os.path.basename(self.audio_path)}...")
        
        sensitivity = self.config["beat_sensitivity"]
        min_beats = self.config["min_beats"]
        
        # Load audio
        y, sr = librosa.load(self.audio_path)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Enhanced beat detection using multiple methods
        if self.config["use_musical_structure"]:
            # Use librosa's beat tracking with tempo estimation
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, 
                                                        trim=False, 
                                                        units='frames')
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
            
            # Also detect onset strength for weak/strong beat classification
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            
            # Combine beat detection with onset detection for more accurate beats
            # This gives better results for many music styles
            all_beats = np.unique(np.concatenate([beat_times, onset_times]))
            
            # Filter out beats that are too close together
            min_beat_interval = 0.2  # Minimum seconds between beats
            filtered_beats = [all_beats[0]]
            for beat in all_beats[1:]:
                if beat - filtered_beats[-1] >= min_beat_interval:
                    filtered_beats.append(beat)
            
            beat_times = np.array(filtered_beats)
            
            # Apply sensitivity scaling
            if sensitivity != 1.0:
                # If sensitivity is high, include more beats; if low, include fewer
                if sensitivity > 1.0:
                    # For high sensitivity, add intermediate beats
                    temp_beats = []
                    for i in range(len(beat_times) - 1):
                        temp_beats.append(beat_times[i])
                        # Maybe add intermediate beats based on sensitivity
                        if (beat_times[i+1] - beat_times[i]) > 2 * min_beat_interval:
                            midpoint = (beat_times[i] + beat_times[i+1]) / 2
                            if random.random() < (sensitivity - 1.0):
                                temp_beats.append(midpoint)
                    temp_beats.append(beat_times[-1])
                    beat_times = np.array(temp_beats)
                else:
                    # For low sensitivity, remove some beats
                    removal_prob = 1.0 - sensitivity
                    temp_beats = [beat_times[0]]
                    for i in range(1, len(beat_times) - 1):
                        if random.random() > removal_prob:
                            temp_beats.append(beat_times[i])
                    temp_beats.append(beat_times[-1])
                    beat_times = np.array(temp_beats)
        else:
            # Simpler beat detection as fallback
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            threshold = np.percentile(onset_env, 75) * sensitivity
            
            # Find peaks with minimum distance
            min_distance = int(sr * 0.2 / 512)  # At least 0.2 seconds between beats
            peaks, _ = find_peaks(onset_env, height=threshold, distance=min_distance)
            beat_times = librosa.frames_to_time(peaks, sr=sr)
        
        # Apply beat grouping if configured
        if self.config["beat_grouping"] > 1:
            # Group every N beats
            grouped_beats = []
            for i in range(0, len(beat_times), self.config["beat_grouping"]):
                if i < len(beat_times):
                    grouped_beats.append(beat_times[i])
            beat_times = np.array(grouped_beats)
        
        # Apply beat offset if configured
        if self.config["beat_offset"] != 0:
            beat_times = beat_times + self.config["beat_offset"]
            # Ensure beats are still within audio duration
            beat_times = beat_times[(beat_times >= 0) & (beat_times <= duration)]
        
        # Ensure minimum number of beats
        if len(beat_times) < min_beats:
            print(f"Too few beats detected ({len(beat_times)}), generating artificial beats...")
            # Generate regular beat intervals based on music duration
            beat_interval = duration / min_beats
            beat_times = np.arange(0, duration, beat_interval)
        
        # Ensure we have beats at the beginning and end
        if beat_times[0] > 0.5:
            beat_times = np.insert(beat_times, 0, 0)
        
        if beat_times[-1] < duration - 0.5:
            beat_times = np.append(beat_times, duration)
        
        self.beat_times = beat_times
        print(f"Detected {len(self.beat_times)} beats")
        
        # Debug: Show the first 10 beat times
        print(f"First 10 beat times: {self.beat_times[:10]}")
        
        return self.beat_times
    
    def _select_clip(self, previous_clip=None, sequence_position=0, total_sequences=1):
        """
        Intelligently select a clip based on configuration settings
        
        Args:
            previous_clip: The previously used clip to avoid repetition
            sequence_position: Position in the sequence (0 to total_sequences-1)
            total_sequences: Total number of sequences in the video
            
        Returns:
            dict containing clip path, speed, and effects information
        """
        available_clips = self.video_clips.copy()
        
        # Apply smart clip selection if enabled
        if self.config["smart_clip_selection"]:
            # Sort clips by usage count (least used first)
            available_clips.sort(key=lambda clip: self.clip_usage_count[clip])
            
            # Remove clips that have reached max repeats
            if self.config["avoid_clip_repetition"]:
                available_clips = [clip for clip in available_clips 
                                  if self.clip_usage_count[clip] < self.config["max_clip_repeats"]]
                
                # If no clips are available, reset the counters and use all clips
                if not available_clips:
                    print("All clips reached maximum usage, resetting counters...")
                    self.clip_usage_count = defaultdict(int)
                    available_clips = self.video_clips.copy()
        
        # Avoid repeating the previous clip if possible
        if previous_clip is not None and previous_clip in available_clips and len(available_clips) > 1:
            available_clips.remove(previous_clip)
        
        # Select a clip
        clip_path = random.choice(available_clips)
        self.clip_usage_count[clip_path] += 1
        
        # Determine if we should apply speed variation
        speed = 1.0  # Default speed
        if self.config["clip_speed_variation"] and random.random() < self.config["speed_change_probability"]:
            # Vary speed based on position in sequence
            if sequence_position < total_sequences * 0.33:
                # Start slower
                speed = random.uniform(self.config["min_speed"], 1.0)
            elif sequence_position > total_sequences * 0.66:
                # End faster
                speed = random.uniform(1.0, self.config["max_speed"])
            else:
                # Middle section - full range
                speed = random.uniform(self.config["min_speed"], self.config["max_speed"])
        
        # Determine visual effect to apply
        visual_effect = "none"
        if self.config["apply_visual_effects"] and random.random() < self.config["visual_effect_probability"]:
            if self.config["transition_variation"]:
                effects = list(self.VISUAL_EFFECTS.keys())
                effects.remove("random")  # Don't include random in the selection
                if "none" in effects:
                    effects.remove("none")  # Don't include "none" in random selection
                visual_effect = random.choice(effects)
            else:
                visual_effect = "none"
        
        # Return clip selection with metadata
        return {
            "path": clip_path,
            "speed": speed,
            "visual_effect": visual_effect,
            "reverse": random.random() < 0.15  # 15% chance of reversed clip
        }
    
    def _select_transition(self, segment_index, total_segments):
        """Select a transition type based on configuration with improved compatibility checks"""
        # Check if we should use a hard cut based on ratio
        if random.random() < self.config["hard_cut_ratio"]:
            return "none"
        
        # If in safe mode, only use the most compatible transitions
        if self.config.get("transition_safe", False):
            safe_transitions = ["none", "crossfade", "fade"]
            
            if self.config["transition_type"] in safe_transitions:
                return self.config["transition_type"]
            else:
                # Use the specified fallback transition or default to crossfade
                return self.config.get("fallback_transition", "crossfade")
            
        # If transition variation is disabled, use the configured transition
        if not self.config["transition_variation"]:
            if self.config["transition_type"] == "random":
                transitions = list(self.TRANSITION_TYPES.keys())
                transitions.remove("random")  # Don't include random in the selection
                return random.choice(transitions)
            else:
                return self.config["transition_type"]
        
        # With variation enabled, select based on position in the video
        transitions = list(self.TRANSITION_TYPES.keys())
        transitions.remove("random")  # Don't include random in the selection
        
        if segment_index < total_segments * 0.25:
            # At the beginning, favor simpler transitions
            weights = [3, 2, 2, 1, 1, 1, 1, 1]
        elif segment_index > total_segments * 0.75:
            # Toward the end, favor more dynamic transitions
            weights = [1, 1, 1, 2, 2, 3, 3, 2]
        else:
            # In the middle, balanced weights
            weights = [2, 2, 2, 2, 2, 2, 2, 2]
        
        # Ensure we have the right number of weights
        weights = weights[:len(transitions)]
        
        # Select transition type based on weights
        return random.choices(transitions, weights=weights, k=1)[0]
    
    def _apply_visual_effect(self, input_file, output_file, effect):
        """Apply a visual effect to a clip"""
        # Define FFmpeg filters for each effect
        effect_filters = {
            "none": "",
            "grayscale": "hue=s=0",
            "sepia": "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131",
            "vibrance": "eq=saturation=1.5:contrast=1.2",
            "vignette": "vignette=PI/4",
            "blur_edges": "vignette=PI/4,gblur=sigma=3:steps=1:planes=0:sigmaV=3",
            "sharpen": "unsharp=5:5:1.5:5:5:0.0",
            "mirror": "hflip",
            "flip": "vflip"
        }
        
        # Get the filter for the requested effect
        filter_str = effect_filters.get(effect, "")
        
        if filter_str:
            cmd = [
                "ffmpeg", "-i", input_file,
                "-vf", filter_str,
                "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                "-an",  # No audio
                output_file, "-y"
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return output_file
        else:
            # If no effect or unknown effect, just copy the file
            shutil.copy(input_file, output_file)
            return output_file
    
    def _apply_speed_effect(self, input_file, output_file, speed):
        """Apply speed adjustment to a clip"""
        if speed == 1.0:
            # No speed change, just copy the file
            shutil.copy(input_file, output_file)
            return output_file
        
        # Use the setpts filter to adjust speed
        # For slower: setpts=PTS*factor (factor > 1)
        # For faster: setpts=PTS/factor (factor > 1)
        if speed < 1.0:
            # Slower
            factor = 1.0 / speed
            filter_str = f"setpts=PTS*{factor}"
        else:
            # Faster
            factor = speed
            filter_str = f"setpts=PTS/{factor}"
        
        cmd = [
            "ffmpeg", "-i", input_file,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            "-an",  # No audio
            output_file, "-y"
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    
    def _create_transition(self, clip1, clip2, output_file, transition_type, duration):
        """Create a transition between two clips with better error handling"""
        # Transition duration in frames (assuming 30fps)
        frames = int(duration * 30)
        
        # First, verify that both clips exist and have valid durations
        if not (os.path.exists(clip1) and os.path.exists(clip2)):
            print(f"  Warning: One of the clip files for transition doesn't exist, falling back to hard cut")
            return self._create_hard_cut(clip1, clip2, output_file)
            
        # Get clip durations to verify transition is possible
        try:
            # Use ffprobe to get clip durations
            get_duration = lambda file: float(subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", file],
                stderr=subprocess.STDOUT, text=True).strip())
            
            clip1_duration = get_duration(clip1)
            clip2_duration = get_duration(clip2)
            
            # Check if clips are long enough for the transition
            if clip1_duration < duration * 0.5 or clip2_duration < duration * 0.5:
                print(f"  Warning: Clips too short for transition ({clip1_duration:.2f}s, {clip2_duration:.2f}s), using shorter transition")
                # Adjust duration to be at most 1/3 of the shortest clip
                duration = min(clip1_duration, clip2_duration) / 3
                if duration < 0.1:  # If still too short, use hard cut
                    print(f"  Warning: Transition duration too short, falling back to hard cut")
                    return self._create_hard_cut(clip1, clip2, output_file)
        except Exception as e:
            print(f"  Warning: Could not check clip durations: {str(e)}, proceeding anyway")
            
        # Define filters for different transition types
        transition_filters = {
            "none": None,  # Hard cut, handled separately
            "crossfade": f"xfade=transition=fade:duration={duration}:offset={duration}",
            "fade": f"xfade=transition=fadeblack:duration={duration}:offset={duration}",
            "wipe_left": f"xfade=transition=wiperight:duration={duration}:offset={duration}",
            "wipe_right": f"xfade=transition=wipeleft:duration={duration}:offset={duration}",
            "zoom_in": f"xfade=transition=zoomin:duration={duration}:offset={duration}",
            "zoom_out": f"xfade=transition=zoomout:duration={duration}:offset={duration}"
        }
        
        filter_str = transition_filters.get(transition_type)
        
        if not filter_str or transition_type == "none":
            return self._create_hard_cut(clip1, clip2, output_file)
        
        # For transitions, use xfade filter
        cmd = [
            "ffmpeg",
            "-i", clip1,
            "-i", clip2,
            "-filter_complex", filter_str,
            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
            output_file, "-y"
        ]
        
        try:
            # Run the command with a timeout to prevent hanging
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
            return output_file
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  Error with {transition_type} transition, trying simpler crossfade instead")
            
            # First attempt: Try a simpler crossfade transition as fallback
            try:
                simple_filter = f"xfade=transition=fade:duration={duration}:offset={duration}"
                simple_cmd = [
                    "ffmpeg",
                    "-i", clip1,
                    "-i", clip2,
                    "-filter_complex", simple_filter,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                    output_file, "-y"
                ]
                subprocess.run(simple_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
                return output_file
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(f"  Simple crossfade also failed, falling back to hard cut")
                return self._create_hard_cut(clip1, clip2, output_file)
    
    def _create_hard_cut(self, clip1, clip2, output_file):
        """Create a hard cut between two clips (fallback method)"""
        # Create a temporary file list for concatenation
        concat_list = os.path.join(self.temp_dir, f"concat_list_{random.randint(1000, 9999)}.txt")
        
        # Check if both clips exist
        if os.path.exists(clip1) and os.path.exists(clip2):
            with open(concat_list, 'w') as f:
                f.write(f"file '{clip1}'\n")
                f.write(f"file '{clip2}'\n")
                
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", concat_list,
                "-c", "copy",
                output_file, "-y"
            ]
        elif os.path.exists(clip1):
            # If only clip1 exists, just use that
            shutil.copy(clip1, output_file)
            return output_file
        elif os.path.exists(clip2):
            # If only clip2 exists, just use that
            shutil.copy(clip2, output_file)
            return output_file
        else:
            # Neither clip exists, return an error
            print(f"  Error: Neither clip exists for hard cut")
            return None
            
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
            return output_file
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            print(f"  Error with hard cut: {str(e)}")
            
            # Last resort - pick one of the clips that exists
            if os.path.exists(clip1):
                shutil.copy(clip1, output_file)
                return output_file
            elif os.path.exists(clip2):
                shutil.copy(clip2, output_file)
                return output_file
            
            return None
    
    def create_beat_synchronized_video(self):
        """
        Create a video synchronized to music beats with enhanced transitions and effects
        """
        if len(self.beat_times) < 2:
            raise ValueError("Not enough beats detected!")
            
        if not self.video_clips:
            raise ValueError("No video clips found!")
        
        # Settings from config
        transition_duration = self.config["transition_duration"]
        min_clip_duration = self.config["min_clip_duration"]
        max_clip_duration = self.config["max_clip_duration"]
        
        # List for all created segments
        created_segments = []
        segments_file = os.path.join(self.temp_dir, "segments.txt")
        
        # Maximum number of segments
        max_segments = min(200, len(self.beat_times) - 1)
        print(f"Creating up to {max_segments} segments...")
        
        # Minimum duration for segments (for safety)
        min_segment_duration = 0.3
        
        # Create separate file for segment list
        with open(segments_file, 'w') as f:
            # Tracking variables
            valid_segments = 0
            previous_clip = None
            
            # Process each beat segment
            for i in range(min(len(self.beat_times) - 1, max_segments)):
                # Beat time points
                start_time = self.beat_times[i]
                end_time = self.beat_times[i+1]
                
                # Calculate duration
                segment_duration = end_time - start_time
                
                # Debug output
                print(f"Beat {i}: from {start_time:.2f}s to {end_time:.2f}s (Duration: {segment_duration:.2f}s)")
                
                # Skip segments that are too short
                if segment_duration < min_segment_duration:
                    print(f"  Skipping: Segment too short ({segment_duration:.2f}s < {min_segment_duration:.2f}s)")
                    continue
                
                # Limit segments that are too long
                if segment_duration > max_clip_duration:
                    segment_duration = max_clip_duration
                    print(f"  Trimming to {segment_duration:.2f}s (maximum duration)")
                
                # Select clip with smart selection
                clip_info = self._select_clip(
                    previous_clip=previous_clip,
                    sequence_position=i,
                    total_sequences=max_segments
                )
                
                clip_path = clip_info["path"]
                speed = clip_info["speed"]
                visual_effect = clip_info["visual_effect"]
                is_reverse = clip_info["reverse"]
                
                # Update tracking variables
                previous_clip = clip_path
                
                # Choose a random start point in the clip
                clip_duration = self.get_clip_duration(clip_path) / speed  # Adjust for speed
                if clip_duration <= segment_duration:
                    clip_start = 0
                else:
                    max_start = clip_duration - segment_duration
                    clip_start = random.uniform(0, max_start)
                
                # Output file for the segment
                base_segment_file = os.path.join(self.temp_dir, f"base_segment_{i:03d}.mp4")
                effect_segment_file = os.path.join(self.temp_dir, f"effect_segment_{i:03d}.mp4")
                speed_segment_file = os.path.join(self.temp_dir, f"speed_segment_{i:03d}.mp4")
                segment_file = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
                
                # Extract the base segment with high quality
                try:
                    # First, extract the raw segment
                    cmd = [
                        "ffmpeg", "-i", clip_path,
                        "-ss", f"{clip_start:.3f}",
                        "-t", f"{segment_duration:.3f}",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                        "-an",  # No audio
                        base_segment_file, "-y"
                    ]
                    
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    # Apply reverse effect if needed
                    if is_reverse:
                        rev_cmd = [
                            "ffmpeg", "-i", base_segment_file,
                            "-vf", "reverse",
                            "-c:v", "libx264", "-preset", "medium", "-crf", "22",
                            effect_segment_file, "-y"
                        ]
                        subprocess.run(rev_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        effect_segment_file = base_segment_file
                    
                    # Apply visual effect if specified
                    if visual_effect != "none":
                        try:
                            effect_segment_file = self._apply_visual_effect(
                                effect_segment_file if is_reverse else base_segment_file,
                                effect_segment_file,
                                visual_effect
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"  Warning: Failed to apply {visual_effect} effect: {str(e)}")
                            effect_segment_file = base_segment_file if not is_reverse else effect_segment_file
                    
                    # Apply speed effect if not 1.0
                    if speed != 1.0:
                        try:
                            speed_segment_file = self._apply_speed_effect(
                                effect_segment_file, 
                                speed_segment_file,
                                speed
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"  Warning: Failed to apply speed effect ({speed}x): {str(e)}")
                            speed_segment_file = effect_segment_file
                    else:
                        speed_segment_file = effect_segment_file
                    
                    # Rename to final segment file
                    shutil.copy(speed_segment_file, segment_file)
                    
                    # Select transition type for next segment
                    next_transition = self._select_transition(i, max_segments)
                    
                    # Add segment info to list
                    if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        f.write(f"file '{segment_file}'\n")
                        
                        # Store metadata about the segment for potential future use
                        segment_info = {
                            "index": i,
                            "file": segment_file,
                            "clip_path": clip_path,
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": segment_duration,
                            "speed": speed,
                            "visual_effect": visual_effect,
                            "is_reverse": is_reverse,
                            "next_transition": next_transition
                        }
                        
                        created_segments.append(segment_info)
                        valid_segments += 1
                        
                        # Log segment details
                        effect_info = f", {visual_effect} effect" if visual_effect != "none" else ""
                        speed_info = f", {speed:.2f}x speed" if speed != 1.0 else ""
                        reverse_info = ", reversed" if is_reverse else ""
                        transition_info = f", next transition: {next_transition}"
                        
                        print(f"Created segment {i+1}/{max_segments}: "
                              f"{os.path.basename(clip_path)}{reverse_info}{speed_info}{effect_info}{transition_info}")
                    else:
                        print(f"  Error: Segment was not created or is empty.")
                        
                except subprocess.CalledProcessError as e:
                    print(f"  Error creating segment {i}: {str(e)}")
                    print(f"  Error output: {e.stderr.decode() if e.stderr else 'No error output'}")
                    continue
            
            print(f"\nSuccessfully created segments: {valid_segments}/{max_segments}")
        
        # Check if segments were created
        if not created_segments:
            raise ValueError("No segments were created! Check beat detection and clip duration.")
        
        # Apply transitions between segments if configured
        if self.config["transition_duration"] > 0 and self.config["transition_type"] != "none":
            print("\nApplying transitions between segments...")
            
            # Create separate file for transitions
            trans_segments_file = os.path.join(self.temp_dir, f"trans_segments_{self.timestamp}.txt")
            
            # Use a counter to ensure unique transition filenames
            trans_counter = 0
            
            with open(trans_segments_file, 'w') as f:
                # Process each segment pair for transitions
                for i in range(len(created_segments) - 1):
                    current_segment = created_segments[i]
                    next_segment = created_segments[i+1]
                    
                    # Output file for the transitional segment
                    trans_file = os.path.join(self.temp_dir, f"trans_{trans_counter:03d}.mp4")
                    trans_counter += 1
                    
                    # Get transition type from current segment's metadata
                    transition_type = current_segment["next_transition"]
                    
                    # If it's a hard cut, just use the original segments
                    if transition_type == "none" or i >= len(created_segments) - 2:
                        f.write(f"file '{current_segment['file']}'\n")
                        if i == len(created_segments) - 2:  # Last pair
                            f.write(f"file '{next_segment['file']}'\n")
                    else:
                        # Create a transition between the segments
                        try:
                            transition_result = self._create_transition(
                                current_segment['file'],
                                next_segment['file'],
                                trans_file,
                                transition_type,
                                self.config["transition_duration"]
                            )
                            
                            if transition_result and os.path.exists(transition_result) and os.path.getsize(transition_result) > 0:
                                f.write(f"file '{transition_result}'\n")
                                print(f"Applied {transition_type} transition between segments {i} and {i+1}")
                            else:
                                # Fallback if transition fails
                                f.write(f"file '{current_segment['file']}'\n")
                                if i == len(created_segments) - 2:  # Last pair
                                    f.write(f"file '{next_segment['file']}'\n")
                                print(f"  Failed to create transition, using segment directly")
                        except Exception as e:
                            # Fallback if transition fails with any exception
                            f.write(f"file '{current_segment['file']}'\n")
                            if i == len(created_segments) - 2:  # Last pair
                                f.write(f"file '{next_segment['file']}'\n")
                            print(f"  Error in transition: {str(e)}")
            
            # Use the transitional segments file instead
            segments_file = trans_segments_file
        
        print(f"\nCombining {len(created_segments)} video segments...")
        
        # Silent output with original audiospur (with timestamp)
        silent_output = os.path.join(self.temp_dir, f"silent_output_{self.timestamp}.mp4")
        
        # Quality presets mapping
        quality_presets = {
            "low": {"preset": "veryfast", "crf": "28"},
            "medium": {"preset": "medium", "crf": "23"},
            "high": {"preset": "slow", "crf": "20"},
            "ultra": {"preset": "slower", "crf": "18"}
        }
        
        # Get quality settings
        quality = self.config["quality"]
        preset = quality_presets.get(quality, quality_presets["high"])
        
        # Concat command with quality settings
        concat_cmd = [
            "ffmpeg", "-f", "concat",
            "-safe", "0",
            "-i", segments_file,
            "-c:v", "libx264", 
            "-preset", preset["preset"], 
            "-crf", preset["crf"],
            silent_output, "-y"
        ]
        
        try:
            print("Executing FFmpeg command:")
            print(" ".join(concat_cmd))
            subprocess.run(concat_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error concatenating segments: {str(e)}")
            # Emergency fallback - try simple concatenation
            print("Trying alternative concatenation method...")
            
            # Create single file for each segment
            alt_segments_file = os.path.join(self.temp_dir, "alt_segments.txt")
            with open(alt_segments_file, 'w') as f:
                for segment in created_segments:
                    segment_file = segment["file"]
                    if os.path.exists(segment_file) and os.path.getsize(segment_file) > 0:
                        f.write(f"file '{os.path.basename(segment_file)}'\n")
            
            # Try with reduced complexity
            alt_cmd = [
                "ffmpeg", "-f", "concat",
                "-safe", "0",
                "-i", alt_segments_file,
                "-c", "copy",
                silent_output, "-y"
            ]
            try:
                subprocess.run(alt_cmd, check=True, cwd=self.temp_dir)
            except subprocess.CalledProcessError:
                print("Alternative method also failed. Trying individual segments...")
                # Extreme fallback - use the first segment
                if len(created_segments) > 0:
                    shutil.copy(created_segments[0]["file"], silent_output)
                else:
                    raise ValueError("No segments could be created.")
        
        # Log the output filename with timestamp
        print(f"\nGenerating final output with filename: {os.path.basename(self.output_file)}")
        
        # Apply audio fade in/out if configured
        audio_fade_filters = []
        if self.config["audio_fade_in"] > 0:
            audio_fade_filters.append(f"afade=t=in:st=0:d={self.config['audio_fade_in']}")
        
        if self.config["audio_fade_out"] > 0:
            # Get audio duration first
            audio_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", self.audio_path
            ]
            result = subprocess.run(audio_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                audio_duration = float(result.stdout.strip())
                fade_start = audio_duration - self.config["audio_fade_out"]
                if fade_start > 0:
                    audio_fade_filters.append(f"afade=t=out:st={fade_start}:d={self.config['audio_fade_out']}")
            except ValueError:
                print("Could not determine audio duration for fade out.")
        
        # Final output with original audio track (with optional fades)
        audio_filter = ",".join(audio_fade_filters)
        
        if audio_filter:
            cmd = [
                "ffmpeg", "-i", silent_output,
                "-i", self.audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-af", audio_filter,
                "-shortest",
                self.output_file, "-y"
            ]
        else:
            cmd = [
                "ffmpeg", "-i", silent_output,
                "-i", self.audio_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                self.output_file, "-y"
            ]
        
        print("\nAdding audio track...")
        subprocess.run(cmd, check=True)
        
        print(f"\nDone! Output saved as: {self.output_file}")
        return self.output_file
    
    def create_config_file(self, config_file=None):
        """Save current configuration to a JSON file with timestamp"""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(self.output_file), f"video_config_{self.timestamp}.json")
            
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
            
        print(f"Configuration saved to: {config_file}")
        return config_file
    
    def load_config_file(self, config_file):
        """Load configuration from a JSON file"""
        with open(config_file, 'r') as f:
            loaded_config = json.load(f)
            
        # Update current configuration
        self.config.update(loaded_config)
        print(f"Configuration loaded from: {config_file}")
        return self.config
    
    def cleanup(self):
        """Delete temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Temporary files deleted: {self.temp_dir}")


def main():
    # Fixed paths
    input_dir = "/Users/dev/womanareoundtheworld/Music_sync/input"
    output_dir = "/Users/dev/womanareoundtheworld/Music_sync/output"
    
    # Check if directories exist
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find audio file in input directory
    audio_files = []
    for ext in ['.mp3', '.wav', '.m4a', '.aac', '.flac']:
        audio_files.extend(glob(os.path.join(input_dir, f'*{ext}')))
    
    if not audio_files:
        print(f"Error: No audio files found in directory: {input_dir}")
        return 1
    
    # Use first audio file found
    audio_file = audio_files[0]
    print(f"Using audio file: {os.path.basename(audio_file)}")
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default output file in output directory with timestamp
    output_file = os.path.join(output_dir, f"music_sync_{timestamp}.mp4")
    temp_dir = os.path.join(output_dir, f"temp_{timestamp}")
    
    # Command-line parameters for additional configuration
    parser = argparse.ArgumentParser(description="Creates a video from clips, synchronized to music rhythm")
    
    # Basic parameters
    parser.add_argument("--output", "-o", type=str, default=output_file,
                        help="Output file path")
    parser.add_argument("--keep-temp", "-k", action="store_true",
                        help="Keep temporary files")
    parser.add_argument("--config", "-c", type=str,
                        help="Load configuration from JSON file")
    
    # Beat detection parameters
    parser.add_argument("--beat-sensitivity", "-b", type=float, default=1.0,
                        help="Beat detection sensitivity (higher = more cuts)")
    parser.add_argument("--beat-offset", type=float, default=0.0,
                        help="Offset beats by this many seconds")
    parser.add_argument("--beat-grouping", type=int, default=1,
                        help="Group every N beats together")
    
    # Clip selection parameters
    parser.add_argument("--min-segment", "-m", type=float, default=1.0,
                        help="Minimum segment length in seconds")
    parser.add_argument("--max-segment", "-M", type=float, default=6.0,
                        help="Maximum segment length in seconds")
    parser.add_argument("--no-speed-variation", action="store_true",
                        help="Disable clip speed variation")
    parser.add_argument("--min-speed", type=float, default=0.5,
                        help="Minimum clip speed (0.5 = half speed)")
    parser.add_argument("--max-speed", type=float, default=2.0,
                        help="Maximum clip speed (2.0 = double speed)")
    
    # Transition parameters
    parser.add_argument("--transition", "-t", type=str, default="crossfade",
                        choices=["none", "crossfade", "fade", "wipe_left", "wipe_right", 
                                "zoom_in", "zoom_out", "random"],
                        help="Transition type")
    parser.add_argument("--transition-duration", "-d", type=float, default=0.5,
                        help="Transition duration in seconds")
    parser.add_argument("--hard-cut-ratio", type=float, default=0.3,
                        help="Ratio of hard cuts to transitions (0.0-1.0)")
    parser.add_argument("--no-transition-variation", action="store_true",
                        help="Disable variation in transition types")
    # Add more transition options with better compatibility
    parser.add_argument("--transition-safe", action="store_true",
                        help="Use only the most compatible transition types")
    parser.add_argument("--fallback-transition", type=str, default="crossfade",
                        choices=["none", "crossfade", "fade"],
                        help="Fallback transition type for compatibility")
    
    # Visual parameters
    parser.add_argument("--quality", "-q", type=str, default="high",
                        choices=["low", "medium", "high", "ultra"],
                        help="Output video quality")
    parser.add_argument("--visual-effects", action="store_true",
                        help="Apply visual effects to clips")
    
    # Parse args
    args = parser.parse_args()
    
    # Create configuration dictionary
    config = {
        # Beat detection parameters
        "beat_sensitivity": args.beat_sensitivity,
        "beat_offset": args.beat_offset,
        "beat_grouping": args.beat_grouping,
        
        # Clip selection parameters
        "min_clip_duration": args.min_segment,
        "max_clip_duration": args.max_segment,
        "clip_speed_variation": not args.no_speed_variation,
        "min_speed": args.min_speed,
        "max_speed": args.max_speed,
        
        # Transition parameters
        "transition_type": args.transition,
        "transition_duration": args.transition_duration,
        "hard_cut_ratio": args.hard_cut_ratio,
        "transition_variation": not args.no_transition_variation,
        "transition_safe": args.transition_safe,
        "fallback_transition": args.fallback_transition,
        
        # Visual parameters
        "quality": args.quality,
        "apply_visual_effects": args.visual_effects,
    }
    
    # Initialize editor
    editor = EnhancedRhythmicVideoEditor(
        clips_dir=input_dir,
        audio_path=audio_file,
        output_file=args.output,
        temp_dir=temp_dir,
        config=config
    )
    
    # If configuration file is specified, load it
    if args.config and os.path.exists(args.config):
        editor.load_config_file(args.config)
    
    try:
        # Run process
        editor.find_video_clips()
        editor.detect_beats()
        output_file = editor.create_beat_synchronized_video()
        
        # Save configuration for future reference
        editor.create_config_file()
        
        if not args.keep_temp:
            editor.cleanup()
            
        print(f"\nSuccessfully completed!\nVideo saved as: {output_file}")
        
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())