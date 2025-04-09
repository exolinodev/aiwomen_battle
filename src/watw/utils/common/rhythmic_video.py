"""
Rhythmic video processing module for the Women Around The World project.

This module provides functionality for creating rhythmically synchronized videos
with both hard cuts and smooth transitions, integrating with the main video editor.
"""

import os
import subprocess
import numpy as np
import random
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple
from dataclasses import dataclass
import librosa
from scipy.signal import find_peaks
import shutil
import logging

from .media_utils import VideoFile, AudioFile, VideoEditor, VideoOperationError

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class BeatInfo:
    """Information about detected beats in audio."""
    times: np.ndarray
    tempo: float
    strength: np.ndarray
    intervals: np.ndarray

class RhythmicVideoEditor(VideoEditor):
    """
    Extended video editor that adds rhythmic synchronization capabilities.
    """
    
    def __init__(self, temp_dir: Optional[Union[str, Path]] = None):
        """
        Initialize a RhythmicVideoEditor instance.
        
        Args:
            temp_dir: Directory for temporary files
        """
        super().__init__(temp_dir)
        self.beat_info: Optional[BeatInfo] = None
    
    def detect_beats(
        self,
        audio_path: Union[str, Path],
        sensitivity: float = 1.2,
        min_beats: int = 10,
        min_bpm: float = 60.0,
        max_bpm: float = 180.0
    ) -> BeatInfo:
        """
        Detect beats in an audio file using multiple methods.
        
        Args:
            audio_path: Path to the audio file
            sensitivity: Sensitivity multiplier for beat detection
            min_beats: Minimum number of beats to detect
            min_bpm: Minimum tempo in beats per minute
            max_bpm: Maximum tempo in beats per minute
            
        Returns:
            BeatInfo: Information about detected beats
            
        Raises:
            VideoOperationError: If beat detection fails
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise VideoOperationError(f"Audio file does not exist: {audio_path}")
        
        logger.info(f"Analyzing audio file: {audio_path.name}")
        
        try:
            # Load audio with increased precision
            y, sr = librosa.load(str(audio_path), sr=44100)
            
            # Combine multiple beat detection methods for better results
            beats = []
            strengths = []
            
            # 1. Onset detection
            logger.info("Method 1: Onset detection...")
            onset_env = librosa.onset.onset_strength(
                y=y, 
                sr=sr,
                hop_length=512,
                aggregate=np.median
            )
            
            # Dynamic threshold based on percentiles
            threshold = np.percentile(onset_env, 75) * sensitivity
            peaks, peak_strengths = find_peaks(
                onset_env, 
                height=threshold,
                distance=sr/512/4  # Minimum distance between peaks: 1/4 beat at 120 BPM
            )
            beat_times_1 = librosa.frames_to_time(peaks, sr=sr, hop_length=512)
            strengths.extend(peak_strengths['peak_heights'])
            
            # 2. Tempogram-based beat detection
            logger.info("Method 2: Tempogram analysis...")
            tempo, beats_frames = librosa.beat.beat_track(
                y=y, 
                sr=sr,
                hop_length=512,
                tightness=100,
                start_bpm=120,
                trim=True
            )
            beat_times_2 = librosa.frames_to_time(beats_frames, sr=sr, hop_length=512)
            strengths.extend([1.0] * len(beat_times_2))  # Default strength for tempogram beats
            
            # 3. Spectral flux-based beat detection
            logger.info("Method 3: Spectral flux analysis...")
            spec = np.abs(librosa.stft(y, hop_length=512))
            spec_flux = np.sum(np.diff(spec, axis=0), axis=0)
            spec_flux = np.maximum(spec_flux, 0.0)
            
            # Normalize and apply moving average
            spec_flux = spec_flux / np.max(spec_flux)
            window_size = 5
            weights = np.hamming(window_size)
            spec_flux_smooth = np.convolve(spec_flux, weights/weights.sum(), mode='same')
            
            # Threshold based on smoothed spectral flux
            sf_threshold = np.percentile(spec_flux_smooth, 75) * sensitivity
            sf_peaks, sf_strengths = find_peaks(spec_flux_smooth, height=sf_threshold, distance=sr/512/4)
            beat_times_3 = librosa.frames_to_time(sf_peaks, sr=sr, hop_length=512)
            strengths.extend(sf_strengths['peak_heights'])
            
            # 4. RMS energy-based beat detection (good for bass beats)
            logger.info("Method 4: Energy-based detection...")
            rms = librosa.feature.rms(y=y, hop_length=512)[0]
            rms_threshold = np.percentile(rms, 75) * sensitivity
            rms_peaks, rms_strengths = find_peaks(rms, height=rms_threshold, distance=sr/512/4)
            beat_times_4 = librosa.frames_to_time(rms_peaks, sr=sr, hop_length=512)
            strengths.extend(rms_strengths['peak_heights'])
            
            # Intelligent combination of all methods
            logger.info("Combining results from all methods...")
            
            # Collect all beats in a list
            all_beats = np.concatenate([beat_times_1, beat_times_2, beat_times_3, beat_times_4])
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
                beat_intervals = np.diff(beat_times)
                
                # Estimate dominant tempo from intervals (median)
                avg_beat_length = np.median(beat_intervals)
                tempo = 60.0 / avg_beat_length  # Convert to BPM
                
                # Add beats where there are large gaps (over 2x average beat length)
                enhanced_beats = [beat_times[0]]
                enhanced_strengths = [beat_strengths[0]]
                
                for i in range(1, len(beat_times)):
                    current_diff = beat_times[i] - beat_times[i-1]
                    if current_diff > 2.2 * avg_beat_length:
                        # Add synthetic beats in the gap
                        num_missing = int(current_diff / avg_beat_length) - 1
                        for j in range(1, num_missing + 1):
                            synthetic_beat = beat_times[i-1] + j * avg_beat_length
                            enhanced_beats.append(synthetic_beat)
                            # Interpolate strength for synthetic beats
                            strength = np.interp(
                                synthetic_beat,
                                [beat_times[i-1], beat_times[i]],
                                [beat_strengths[i-1], beat_strengths[i]]
                            )
                            enhanced_strengths.append(strength)
                    
                    enhanced_beats.append(beat_times[i])
                    enhanced_strengths.append(beat_strengths[i])
                
                beat_times = np.array(enhanced_beats)
                beat_strengths = np.array(enhanced_strengths)
            
            # Ensure we have enough beats
            if len(beat_times) < min_beats:
                logger.warning(f"Too few beats detected ({len(beat_times)}), generating synthetic beats...")
                # Determine a musically sensible tempo
                duration = librosa.get_duration(y=y, sr=sr)
                
                if tempo < min_bpm or tempo > max_bpm:
                    tempo = 120  # Default tempo
                
                beat_interval = 60.0 / tempo  # Convert BPM to seconds
                beat_times = np.arange(0, duration, beat_interval)
                beat_strengths = np.ones_like(beat_times)  # Default strength for synthetic beats
            
            # Add start and end if needed
            if beat_times[0] > 0.5:
                beat_times = np.insert(beat_times, 0, 0)
                beat_strengths = np.insert(beat_strengths, 0, 1.0)
            
            duration = librosa.get_duration(y=y, sr=sr)
            if beat_times[-1] < duration - 0.5:
                beat_times = np.append(beat_times, duration)
                beat_strengths = np.append(beat_strengths, 1.0)
            
            # Sort and remove duplicates
            sort_idx = np.argsort(beat_times)
            beat_times = beat_times[sort_idx]
            beat_strengths = beat_strengths[sort_idx]
            
            # Calculate beat intervals
            beat_intervals = np.diff(beat_times)
            
            # Create BeatInfo object
            self.beat_info = BeatInfo(
                times=beat_times,
                tempo=tempo,
                strength=beat_strengths,
                intervals=beat_intervals
            )
            
            logger.info(f"Detected {len(beat_times)} beats at {tempo:.1f} BPM")
            return self.beat_info
            
        except Exception as e:
            raise VideoOperationError(f"Beat detection failed: {str(e)}")
    
    def create_rhythmic_video(
        self,
        video_clips_dir: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        min_clip_duration: float = 0.5,
        max_clip_duration: float = 5.0,
        transition_duration: float = 0.5,
        reverse_probability: float = 0.3,
        background_music_path: Optional[Union[str, Path]] = None,
        music_volume: float = 0.1
    ) -> VideoFile:
        """
        Create a rhythmically synchronized video with hard cuts.
        
        Args:
            video_clips_dir: Directory containing video clips
            audio_path: Path to the audio file
            output_path: Path to save the final video
            min_clip_duration: Minimum duration for each clip
            max_clip_duration: Maximum duration for each clip
            transition_duration: Duration of transitions between clips
            reverse_probability: Probability of playing a clip in reverse
            background_music_path: Optional path to background music
            music_volume: Volume of background music (0.0 to 1.0)
            
        Returns:
            VideoFile: The final video file
            
        Raises:
            VideoOperationError: If video creation fails
        """
        video_clips_dir = Path(video_clips_dir)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        # Validate inputs
        if not video_clips_dir.exists():
            raise VideoOperationError(f"Video clips directory does not exist: {video_clips_dir}")
        if not audio_path.exists():
            raise VideoOperationError(f"Audio file does not exist: {audio_path}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find video clips
        video_clips = []
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv']:
            video_clips.extend(list(video_clips_dir.glob(f'*{ext}')))
        
        if not video_clips:
            raise VideoOperationError(f"No video clips found in {video_clips_dir}")
        
        # Detect beats in audio
        beat_info = self.detect_beats(audio_path)
        
        # Create enhanced clip list with variations
        enhanced_clips = []
        for clip in video_clips:
            enhanced_clips.append({"path": clip, "reverse": False})
            enhanced_clips.append({"path": clip, "reverse": True})
        
        # Shuffle multiple times for better mixing
        for _ in range(3):
            random.shuffle(enhanced_clips)
        
        # Track last used clip
        last_clip_path = None
        last_was_reverse = False
        used_count = {str(clip): 0 for clip in video_clips}
        
        # Process each beat segment
        segments = []
        for i in range(len(beat_info.times) - 1):
            # Beat times
            start_time = beat_info.times[i]
            end_time = beat_info.times[i+1]
            
            # Consider transition duration
            duration = (end_time - start_time) + transition_duration
            
            # Skip too short segments, limit too long ones
            if duration < min_clip_duration:
                continue
            if duration > max_clip_duration:
                duration = max_clip_duration
            
            # Intelligent clip selection
            valid_clips = [
                clip for clip in enhanced_clips
                if clip["path"] != last_clip_path or clip["reverse"] != last_was_reverse
            ]
            
            if not valid_clips:
                valid_clips = enhanced_clips
            
            # Choose clip with preference for less used ones
            chosen_clip = min(
                valid_clips,
                key=lambda x: used_count[str(x["path"])]
            )
            
            clip_path = chosen_clip["path"]
            is_reverse = chosen_clip["reverse"]
            
            # Update tracking variables
            last_clip_path = clip_path
            last_was_reverse = is_reverse
            used_count[str(clip_path)] += 1
            
            # Create segment
            segment_path = self.temp_dir / f"segment_{i:03d}.mp4"
            
            try:
                # Trim video to duration
                video = VideoFile(clip_path)
                if is_reverse:
                    # For reverse clips, we need to extract the full clip first
                    temp_path = self.temp_dir / f"temp_{i:03d}.mp4"
                    video = video.trim(0, duration, temp_path)
                    
                    # Then reverse it
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(video.path),
                        '-vf', 'reverse',
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '22',
                        str(segment_path)
                    ]
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Clean up temp file
                    if temp_path.exists():
                        temp_path.unlink()
                else:
                    # For forward clips, just trim
                    video = video.trim(0, duration, segment_path)
                
                segments.append(segment_path)
                logger.info(f"Created segment {i+1}: {clip_path.name} "
                          f"({'reverse' if is_reverse else 'forward'})")
                
            except Exception as e:
                logger.error(f"Failed to create segment {i}: {str(e)}")
                continue
        
        if not segments:
            raise VideoOperationError("No segments were created")
        
        # Concatenate segments
        try:
            # Create concat list file
            concat_list_path = self.temp_dir / "concat_list.txt"
            with open(concat_list_path, 'w') as f:
                for segment in segments:
                    f.write(f"file '{segment}'\n")
            
            # Concatenate videos
            silent_output = self.temp_dir / "silent_output.mp4"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_list_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '22',
                str(silent_output)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Combine with audio
            if background_music_path and Path(background_music_path).exists():
                # Mix audio files
                mixed_audio = self.temp_dir / "mixed_audio.mp3"
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(audio_path),
                    '-i', str(background_music_path),
                    '-filter_complex', f'[1:a]volume={music_volume}[a1];[0:a][a1]amix=inputs=2:duration=first',
                    '-acodec', 'libmp3lame',
                    '-q:a', '2',
                    str(mixed_audio)
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # Combine video with mixed audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(silent_output),
                    '-i', str(mixed_audio),
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    str(output_path)
                ]
            else:
                # Combine video with original audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(silent_output),
                    '-i', str(audio_path),
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    str(output_path)
                ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            return VideoFile(output_path)
            
        except Exception as e:
            raise VideoOperationError(f"Failed to create final video: {str(e)}")
    
    def create_smooth_video(
        self,
        video_clips_dir: Union[str, Path],
        audio_path: Union[str, Path],
        output_path: Union[str, Path],
        segment_duration: float = 3.0,
        transition_duration: float = 0.5,
        reverse_probability: float = 0.3,
        background_music_path: Optional[Union[str, Path]] = None,
        music_volume: float = 0.1
    ) -> VideoFile:
        """
        Create a music-synchronized video with smooth transitions.
        
        Args:
            video_clips_dir: Directory containing video clips
            audio_path: Path to the audio file
            output_path: Path to save the final video
            segment_duration: Duration of each segment
            transition_duration: Duration of transitions between segments
            reverse_probability: Probability of playing a clip in reverse
            background_music_path: Optional path to background music
            music_volume: Volume of background music (0.0 to 1.0)
            
        Returns:
            VideoFile: The final video file
            
        Raises:
            VideoOperationError: If video creation fails
        """
        video_clips_dir = Path(video_clips_dir)
        audio_path = Path(audio_path)
        output_path = Path(output_path)
        
        # Validate inputs
        if not video_clips_dir.exists():
            raise VideoOperationError(f"Video clips directory does not exist: {video_clips_dir}")
        if not audio_path.exists():
            raise VideoOperationError(f"Audio file does not exist: {audio_path}")
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Find video clips
        video_clips = []
        for ext in ['.mp4', '.mov', '.avi', '.mkv', '.wmv']:
            video_clips.extend(list(video_clips_dir.glob(f'*{ext}')))
        
        if not video_clips:
            raise VideoOperationError(f"No video clips found in {video_clips_dir}")
        
        # Get audio duration
        audio = AudioFile(audio_path)
        audio_duration = audio.get_duration()
        logger.info(f"Audio duration: {audio_duration:.2f} seconds")
        
        # Create enhanced clip list
        enhanced_clips = []
        for clip in video_clips:
            enhanced_clips.append({"path": clip, "reverse": False})
            enhanced_clips.append({"path": clip, "reverse": True})
        
        # Shuffle multiple times for better mixing
        for _ in range(3):
            random.shuffle(enhanced_clips)
        
        # Track last used clip
        last_clip_path = None
        segments = []
        
        # Calculate number of segments
        effective_segment_duration = segment_duration * 0.8  # Reduce by 20% for transitions
        num_segments = int(audio_duration / effective_segment_duration) + 1
        logger.info(f"Creating {num_segments} segments of {segment_duration:.1f} seconds each")
        
        # Create segments file
        segments_file = self.temp_dir / "segments.txt"
        with open(segments_file, 'w') as f:
            for i in range(num_segments):
                # Choose a clip that's not the last one
                valid_clips = [
                    clip for clip in enhanced_clips
                    if clip["path"] != last_clip_path
                ]
                
                if not valid_clips:
                    valid_clips = enhanced_clips
                
                chosen_clip = random.choice(valid_clips)
                clip_path = chosen_clip["path"]
                is_reverse = chosen_clip["reverse"]
                
                # Update last clip
                last_clip_path = clip_path
                
                # Determine random start point in clip
                video = VideoFile(clip_path)
                clip_duration = video.get_metadata().duration
                
                if clip_duration <= segment_duration:
                    clip_start = 0
                else:
                    max_start = clip_duration - segment_duration
                    clip_start = random.uniform(0, max_start)
                
                # Create segment
                segment_path = self.temp_dir / f"segment_{i:03d}.mp4"
                
                try:
                    # Extract segment
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', str(clip_path),
                        '-ss', f"{clip_start:.3f}",
                        '-t', f"{segment_duration:.3f}",
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '22',
                        '-an'  # No audio
                    ]
                    
                    if is_reverse:
                        cmd += ['-vf', 'reverse']
                    
                    cmd += [str(segment_path)]
                    
                    subprocess.run(cmd, check=True, capture_output=True)
                    
                    # Add to segments list
                    if segment_path.exists() and segment_path.stat().st_size > 0:
                        f.write(f"file '{segment_path}'\n")
                        segments.append(segment_path)
                        logger.info(f"Created segment {i+1}: {clip_path.name} "
                                  f"({'reverse' if is_reverse else 'forward'})")
                    else:
                        logger.error(f"Segment {i} was not created or is empty")
                        
                except Exception as e:
                    logger.error(f"Failed to create segment {i}: {str(e)}")
                    continue
        
        if not segments:
            raise VideoOperationError("No segments were created")
        
        # Concatenate segments
        try:
            silent_output = self.temp_dir / "silent_output.mp4"
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(segments_file),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '22',
                str(silent_output)
            ]
            
            subprocess.run(cmd, check=True)
            
            # Combine with audio
            if background_music_path and Path(background_music_path).exists():
                # Mix audio files
                mixed_audio = self.temp_dir / "mixed_audio.mp3"
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(audio_path),
                    '-i', str(background_music_path),
                    '-filter_complex', f'[1:a]volume={music_volume}[a1];[0:a][a1]amix=inputs=2:duration=first',
                    '-acodec', 'libmp3lame',
                    '-q:a', '2',
                    str(mixed_audio)
                ]
                subprocess.run(cmd, check=True)
                
                # Combine video with mixed audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(silent_output),
                    '-i', str(mixed_audio),
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    str(output_path)
                ]
            else:
                # Combine video with original audio
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(silent_output),
                    '-i', str(audio_path),
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-b:a', '192k',
                    '-shortest',
                    str(output_path)
                ]
            
            subprocess.run(cmd, check=True)
            
            return VideoFile(output_path)
            
        except Exception as e:
            raise VideoOperationError(f"Failed to create final video: {str(e)}") 