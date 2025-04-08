#!/usr/bin/env python3
import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, play

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from Generation.render import (
        check_credentials,
        generate_base_image_tensorart,
        generate_animation_runway,
    )
    from Generation.video_editor import combine_video_with_voiceover, concatenate_videos
    from Generation.countries import COUNTRIES
    from Generation.prompts import ANIMATION_TEMPLATE
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all required modules are in the correct directory.")
    sys.exit(1)

def load_config():
    """Load the configuration from config.json."""
    try:
        config_path = os.path.join("Generation", "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Successfully loaded configuration with {len(config)} countries.")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return None

def test_elevenlabs_connection():
    """Test the connection to ElevenLabs API and generate a simple voice-over."""
    try:
        # Load environment variables
        env_path = Path("Generation/.env")
        load_dotenv(dotenv_path=env_path)
        
        # Get API key
        api_key = os.getenv("ELEVENLABS_API_SECRET")
        if not api_key:
            print("Error: ELEVENLABS_API_SECRET not found in .env file")
            return False
        
        # Set API key
        set_api_key(api_key)
        print("Successfully connected to ElevenLabs API")
        
        # Generate a simple test audio
        print("Generating test audio...")
        audio = generate(
            text="This is a test of the ElevenLabs voice-over integration.",
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_voiceover.mp3")
        save(audio, output_path)
        print(f"Test audio saved to: {output_path}")
        
        # Play the audio
        print("Playing test audio...")
        play(audio)
        
        return True
    
    except Exception as e:
        print(f"Error testing ElevenLabs integration: {e}")
        return False

def test_complete_workflow():
    """Test the complete workflow from generation to final video."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            return False
            
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = f"test_workflow_{timestamp}"
        
        # Directory for generated animation clips
        generated_clips_dir = os.path.join(output_base_dir, "generated_clips")
        # Directory for temporary files (animator base images)
        temp_files_dir = os.path.join(output_base_dir, "temp_files")
        # Directory for final video
        final_video_dir = os.path.join(output_base_dir, "final_video")
        # Directory for voice-over audio files
        voiceover_dir = os.path.join(output_base_dir, "voiceovers")
        
        # Create directories
        os.makedirs(generated_clips_dir, exist_ok=True)
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(final_video_dir, exist_ok=True)
        os.makedirs(voiceover_dir, exist_ok=True)
        print(f"Created directory structure in: {output_base_dir}")
        
        # --- Stage 1: Base Image Generation ---
        print("\n" + "="*20 + " Stage 1: Generating Base Images " + "="*20)
        if not check_credentials():
            print("Error: API credentials not valid")
            return False
        
        # Generate base images for all countries
        base_image_paths = []
        for country_key, country_data in config.items():
            print(f"Generating base image for: {country_data['name']}")
            
            # Generate base image
            base_image_result = generate_base_image_tensorart(
                output_directory=temp_files_dir,
                prompt_text=country_data['base_image_prompt'],
                prompt_id=country_data['id']
            )
            
            if not base_image_result or not base_image_result[0] or not os.path.exists(base_image_result[0]):
                print(f"Error: Failed to generate base image for {country_data['name']}")
                return False
            
            base_image_path = base_image_result[0]
            base_image_paths.append(base_image_path)
            print(f"Base image for {country_data['name']} saved to: {base_image_path}")
        
        # --- Stage 2: Animation Generation ---
        print("\n" + "="*20 + " Stage 2: Generating Animations " + "="*20)
        
        # Generate animations for all countries
        animation_paths = []
        for i, (country_key, country_data) in enumerate(config.items()):
            # Format the animation prompt using the template
            animation_prompt = ANIMATION_TEMPLATE.format(
                name=country_data['name'],
                colors=country_data['colors'],
                flag_description=country_data['flag_description']
            )
            
            # Generate animation
            animation_path = generate_animation_runway(
                base_image_path=base_image_paths[i],
                animation_prompt_text=animation_prompt,
                output_directory=generated_clips_dir,
                output_filename_base=f"animation_{country_data['id']}",
                seed=12345 + i  # Different seed for each animation
            )
            
            if not animation_path or not os.path.exists(animation_path):
                print(f"Error: Failed to generate animation for {country_data['name']}")
                return False
            
            animation_paths.append(animation_path)
            print(f"Animation for {country_data['name']} saved to: {animation_path}")
        
        # --- Stage 3: Voice-over Generation ---
        print("\n" + "="*20 + " Stage 3: Generating Voice-overs " + "="*20)
        
        # Load environment variables
        env_path = Path("Generation/.env")
        load_dotenv(dotenv_path=env_path)
        
        # Get API key
        api_key = os.getenv("ELEVENLABS_API_SECRET")
        if not api_key:
            print("Error: ELEVENLABS_API_SECRET not found in .env file")
            return False
        
        # Set API key
        set_api_key(api_key)
        print("Successfully connected to ElevenLabs API")
        
        # Generate voice-overs for all clips
        voiceover_files = []
        countries = list(config.keys())
        for i, country in enumerate(countries, 1):
            country_id = config[country]["id"]
            print(f"Generating voice-over for {country.title()}...")
            
            # Use intro_voiceover for the first country, transition_voiceover for others
            if i == 1:
                script = config[country]["intro_voiceover"]
            else:
                script = config[country]["transition_voiceover"]
                
            # Skip if script is None
            if script is None:
                print(f"No voiceover script defined for {country.title()}. Skipping...")
                # Create a placeholder voiceover file
                voiceover_file = os.path.join(voiceover_dir, f"voiceover_{country_id}.mp3")
                # Create an empty file
                with open(voiceover_file, 'w') as f:
                    f.write("")
                voiceover_files.append(voiceover_file)
                print(f"Created placeholder voiceover file for {country.title()}: {voiceover_file}")
                continue
                
            print(f"Script for {country.title()}: {script}")
            
            # Generate audio
            print(f"Generating audio for {country.title()}...")
            audio = generate(
                text=script,
                voice="Bella",
                model="eleven_multilingual_v2"
            )
            
            # Save the audio to a file
            voiceover_file = os.path.join(voiceover_dir, f"voiceover_{country_id}.mp3")
            save(audio, voiceover_file)
            voiceover_files.append(voiceover_file)
            print(f"Voice-over for {country.title()} saved to: {voiceover_file}")
        
        # --- Stage 4: Video Editing ---
        print("\n" + "="*20 + " Stage 4: Creating Final Video " + "="*20)
        
        # First, combine each animation with its corresponding voice-over
        combined_clips = []
        for i, (animation_path, voiceover_path) in enumerate(zip(animation_paths, voiceover_files)):
            # Output path for the combined clip
            combined_clip_path = os.path.join(final_video_dir, f"combined_clip_{i+1}.mp4")
            
            # Combine video with voice-over
            print(f"Combining clip {i+1} with its voice-over...")
            success = combine_video_with_voiceover(
                video_path=animation_path,
                voiceover_path=voiceover_path,
                output_path=combined_clip_path,
                is_first_clip=(i == 0)  # First clip gets special treatment
            )
            
            if success and os.path.exists(combined_clip_path):
                print(f"Successfully created combined clip {i+1}: {combined_clip_path}")
                combined_clips.append(combined_clip_path)
            else:
                print(f"Failed to create combined clip {i+1}")
                return False
        
        # Now combine all clips into a single final video
        print("\nCombining all clips into a single final video...")
        final_video_path = os.path.join(final_video_dir, "final_video.mp4")
        
        # Create a concat file listing the clips
        concat_file = os.path.join(final_video_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for clip_path in combined_clips:
                f.write(f"file '{os.path.basename(clip_path)}'\n")
        
        print("Created concat file with the following content:")
        with open(concat_file, 'r') as f:
            print(f.read())
        
        # Use FFmpeg to concatenate the clips
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            final_video_path
        ]
        
        try:
            print("Executing FFmpeg concat command...")
            subprocess.run(concat_cmd, check=True, capture_output=True, text=True)
            
            if os.path.exists(final_video_path) and os.path.getsize(final_video_path) > 0:
                print(f"Successfully created final video: {final_video_path}")
                return True
            else:
                print("Failed to create final video: Output file is empty or doesn't exist")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"Error during video concatenation: {e.stderr}")
            return False
        except Exception as e:
            print(f"Unexpected error during video concatenation: {e}")
            return False
    
    except Exception as e:
        print(f"Error in complete workflow test: {e}")
        return False

def test_audio_composition():
    """Test the audio composition with existing animations."""
    try:
        # Load configuration
        config = load_config()
        if not config:
            return False
            
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

        # Process all countries from config
        countries = list(config.keys())
        animation_files = []
        for i, country in enumerate(countries, 1):
            country_id = config[country]["id"]
            animation_file = os.path.join(animation_dir, f"animation_{country_id}_seed{12344 + i}.mp4")
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
            return False
        
        # Set API key
        set_api_key(api_key)
        print("Successfully connected to ElevenLabs API")
        
        # Generate voice-overs for all clips
        voiceover_files = []
        for i, country in enumerate(countries, 1):
            country_id = config[country]["id"]
            print(f"Generating voice-over for {country.title()}...")
            
            # Use intro_voiceover for the first country, transition_voiceover for others
            if i == 1:
                script = config[country]["intro_voiceover"]
            else:
                script = config[country]["transition_voiceover"]
                
            # Skip if script is None
            if script is None:
                print(f"No voiceover script defined for {country.title()}. Skipping...")
                # Create a placeholder voiceover file
                voiceover_file = os.path.join(voiceover_dir, f"voiceover_{country_id}.mp3")
                # Create an empty file
                with open(voiceover_file, 'w') as f:
                    f.write("")
                voiceover_files.append(voiceover_file)
                print(f"Created placeholder voiceover file for {country.title()}: {voiceover_file}")
                continue
                
            print(f"Script for {country.title()}: {script}")
            
            # Generate audio
            print(f"Generating audio for {country.title()}...")
            audio = generate(
                text=script,
                voice="Bella",
                model="eleven_multilingual_v2"
            )
            
            # Save the audio to a file
            voiceover_file = os.path.join(voiceover_dir, f"voiceover_{country_id}.mp3")
            save(audio, voiceover_file)
            voiceover_files.append(voiceover_file)
            print(f"Voice-over for {country.title()} saved to: {voiceover_file}")

        print("\n==================== Stage 2: Creating Final Video ====================")
        # Combine each animation with its voice-over
        combined_clips = []
        for i, (animation_file, voiceover_file) in enumerate(zip(animation_files, voiceover_files), 1):
            print(f"Combining clip {i} with its voice-over...")
            combined_clip = os.path.join(final_video_dir, f"combined_clip_{i}.mp4")
            # Pass is_first_clip=True for first clip
            if combine_video_with_voiceover(animation_file, voiceover_file, combined_clip, is_first_clip=(i == 1)):
                combined_clips.append(combined_clip)
            else:
                print(f"Failed to create combined clip {i}")
                return False

        # Create list file for concatenation
        concat_list_path = os.path.join(final_video_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for clip in combined_clips:
                f.write(f"file '{os.path.basename(clip)}'\n")

        # Concatenate all clips into final video
        final_video_path = os.path.join(final_video_dir, "final_video.mp4")
        if concatenate_videos(concat_list_path, final_video_path, final_video_dir):
            print("Successfully created final video!")
            return True
        else:
            print("Failed to create final video")
            return False

    except Exception as e:
        print(f"Error in audio and composition test: {e}")
        return False

def test_video_editor():
    """Test the video editor to combine a video with a voice-over."""
    try:
        # Create test directories
        test_dir = "test_video_editor"
        os.makedirs(test_dir, exist_ok=True)
        
        # Check if we have a test video and voice-over
        test_video_path = os.path.join("test_output", "test_video.mp4")
        test_voiceover_path = os.path.join("test_output", "voiceover_1.mp3")
        
        # If test video doesn't exist, create a dummy video
        if not os.path.exists(test_video_path):
            print("Test video not found. Creating a dummy video...")
            # Create a simple video using FFmpeg
            os.system(f"ffmpeg -y -f lavfi -i color=c=blue:s=1280x720:d=5 -c:v libx264 {test_video_path}")
        
        # Check if voice-over exists
        if not os.path.exists(test_voiceover_path):
            print("Voice-over not found. Please run test_voiceover_only.py first.")
            return False
        
        # Output path for the combined video
        output_path = os.path.join(test_dir, "combined_video.mp4")
        
        # Combine video with voice-over
        print(f"Combining video with voice-over...")
        success = combine_video_with_voiceover(
            video_path=test_video_path,
            voiceover_path=test_voiceover_path,
            output_path=output_path
        )
        
        if success and os.path.exists(output_path):
            print(f"Successfully combined video with voice-over: {output_path}")
            return True
        else:
            print("Failed to combine video with voice-over")
            return False
    
    except Exception as e:
        print(f"Error testing video editor: {e}")
        return False

def print_menu():
    """Print the menu of available tests."""
    print("\n" + "="*50)
    print("WOMEN AROUND THE WORLD - TEST MENU")
    print("="*50)
    print("1. Test ElevenLabs Connection")
    print("2. Test Video Editor")
    print("3. Test Audio Composition (with existing animations)")
    print("4. Test Complete Workflow (generate everything)")
    print("5. Exit")
    print("="*50)
    print("Note: Edit Generation/config.json to manage the countries in the final output")
    print("="*50)

def main():
    """Main function to run the tests."""
    while True:
        print_menu()
        choice = input("Enter your choice (1-5): ")
        
        if choice == "1":
            print("\n" + "="*20 + " Testing ElevenLabs Connection " + "="*20)
            if test_elevenlabs_connection():
                print("\nElevenLabs connection test passed!")
            else:
                print("\nElevenLabs connection test failed!")
        
        elif choice == "2":
            print("\n" + "="*20 + " Testing Video Editor " + "="*20)
            if test_video_editor():
                print("\nVideo editor test passed!")
            else:
                print("\nVideo editor test failed!")
        
        elif choice == "3":
            print("\n" + "="*20 + " Testing Audio Composition " + "="*20)
            if test_audio_composition():
                print("\nAudio composition test passed!")
            else:
                print("\nAudio composition test failed!")
        
        elif choice == "4":
            print("\n" + "="*20 + " Testing Complete Workflow " + "="*20)
            if test_complete_workflow():
                print("\nComplete workflow test passed!")
            else:
                print("\nComplete workflow test failed!")
        
        elif choice == "5":
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 