#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key
from Generation.video_editor import combine_video_with_voiceover, concatenate_videos

def main():
    # Use existing animations from test_workflow
    animation_dir = "test_workflow_20250407_152531/generated_clips"
    print(f"Testing voice-over generation and video composition using animations from: {animation_dir}")

    # Create output directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"test_audio_{timestamp}"
    voiceover_dir = os.path.join(output_dir, "voiceovers")
    final_video_dir = os.path.join(output_dir, "final_video")
    os.makedirs(voiceover_dir, exist_ok=True)
    os.makedirs(final_video_dir, exist_ok=True)
    print(f"Created directory structure in: {output_dir}")

    # Load configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # Process all countries
    countries = ["japan", "france", "italy", "spain", "germany"]
    animation_files = []
    for i, country in enumerate(countries, 1):
        animation_file = os.path.join(animation_dir, f"animation_{i:02d}_{country}_seed{12344 + i}.mp4")
        if os.path.exists(animation_file):
            print(f"Found animation for {country.title()}: {animation_file}")
            animation_files.append(animation_file)

    # --- Stage 1: Voice-over Generation ---
    print("\n==================== Stage 1: Generating Voice-overs ====================")
    
    # Load environment variables
    env_path = Path("Generation/.env")
    load_dotenv(dotenv_path=env_path)
    
    # Get API key
    api_key = os.getenv("ELEVENLABS_API_SECRET")
    if not api_key:
        print("Error: ELEVENLABS_API_SECRET not found in .env file")
        return
    
    # Set API key
    set_api_key(api_key)
    print("Successfully connected to ElevenLabs API")
    
    # Generate voice-overs for all clips
    voiceover_files = []
    for i, country in enumerate(countries, 1):
        print(f"Generating voice-over for {country.title()}...")
        script = config[country]["voice_over_text"]
        print(f"Script for {country.title()}: {script}")
        
        # Generate audio
        print(f"Generating audio for {country.title()}...")
        audio = generate(
            text=script,
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        voiceover_file = os.path.join(voiceover_dir, f"voiceover_{i:02d}_{country}.mp3")
        save(audio, voiceover_file)
        voiceover_files.append(voiceover_file)
        print(f"Voice-over for {country.title()} saved to: {voiceover_file}")

    print("\n==================== Stage 2: Creating Final Video ====================")
    # Combine each animation with its voice-over
    combined_clips = []
    for i, (animation_file, voiceover_file) in enumerate(zip(animation_files, voiceover_files), 1):
        print(f"Combining clip {i} with its voice-over...")
        combined_clip = os.path.join(final_video_dir, f"combined_clip_{i}.mp4")
        # Pass is_first_clip=True for Japan (first clip)
        if combine_video_with_voiceover(animation_file, voiceover_file, combined_clip, is_first_clip=(i == 1)):
            combined_clips.append(combined_clip)
        else:
            print(f"Failed to create combined clip {i}")
            return

    # Create list file for concatenation
    concat_list_path = os.path.join(final_video_dir, "concat_list.txt")
    with open(concat_list_path, "w") as f:
        for clip in combined_clips:
            f.write(f"file '{os.path.basename(clip)}'\n")

    # Concatenate all clips into final video
    final_video_path = os.path.join(final_video_dir, "final_video.mp4")
    if concatenate_videos(concat_list_path, final_video_path, final_video_dir):
        print("Successfully created final video!")
    else:
        print("Failed to create final video")
        return

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error in audio and composition test: {e}")
        print("\nAudio and composition test failed!") 