#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, play
import subprocess

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from Generation.render import (
        check_credentials,
        generate_base_image_tensorart,
        generate_animation_runway,
    )
    from Generation.video_editor import combine_video_with_voiceover
    from Generation.countries import COUNTRIES
    from Generation.prompts import ANIMATION_TEMPLATE
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure all required modules are in the correct directory.")
    sys.exit(1)

def test_complete_workflow():
    """Test the complete workflow with two countries and different voiceovers."""
    try:
        # Create timestamped output directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
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
        
        # Generate base images for both countries
        base_image_paths = []
        for country_key, country_data in COUNTRIES.items():
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
        
        # Generate animations for both countries
        animation_paths = []
        for i, (country_key, country_data) in enumerate(COUNTRIES.items()):
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
        
        # Generate voice-overs for both clips
        voiceover_paths = []
        video_number = 1
        
        # First voice-over: Intro with question and first country
        print("Generating voice-over for first clip (Intro)...")
        first_country = list(COUNTRIES.values())[0]
        script1 = (
            f"Video #{video_number}. What country do you like most? "
            f"Let me show you {first_country['name']} first."
        )
        
        print(f"Script 1: {script1}")
        
        # Generate audio for first clip
        print("Generating audio for first clip...")
        audio1 = generate(
            text=script1,
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        voiceover_path1 = os.path.join(voiceover_dir, f"voiceover_{first_country['id']}.mp3")
        save(audio1, voiceover_path1)
        voiceover_paths.append(voiceover_path1)
        print(f"Voice-over for first clip saved to: {voiceover_path1}")
        
        # Second voice-over: Second country introduction
        print("Generating voice-over for second clip (Country name)...")
        second_country = list(COUNTRIES.values())[1]
        script2 = f"And now, {second_country['name']}."
        
        print(f"Script 2: {script2}")
        
        # Generate audio for second clip
        print("Generating audio for second clip...")
        audio2 = generate(
            text=script2,
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        voiceover_path2 = os.path.join(voiceover_dir, f"voiceover_{second_country['id']}.mp3")
        save(audio2, voiceover_path2)
        voiceover_paths.append(voiceover_path2)
        print(f"Voice-over for second clip saved to: {voiceover_path2}")
        
        # --- Stage 4: Video Editing ---
        print("\n" + "="*20 + " Stage 4: Creating Final Video " + "="*20)
        
        # First, combine each animation with its corresponding voice-over
        combined_clips = []
        for i, (animation_path, voiceover_path) in enumerate(zip(animation_paths, voiceover_paths)):
            # Output path for the combined clip
            combined_clip_path = os.path.join(final_video_dir, f"combined_clip_{i+1}.mp4")
            
            # Combine video with voice-over
            print(f"Combining clip {i+1} with its voice-over...")
            success = combine_video_with_voiceover(
                video_path=animation_path,
                voiceover_path=voiceover_path,
                output_path=combined_clip_path
            )
            
            if success and os.path.exists(combined_clip_path):
                print(f"Successfully created combined clip {i+1}: {combined_clip_path}")
                combined_clips.append(combined_clip_path)
            else:
                print(f"Failed to create combined clip {i+1}")
                return False
        
        # Now combine the two clips into a single final video
        print("\nCombining all clips into a single final video...")
        final_video_path = os.path.join(final_video_dir, "final_video.mp4")
        
        # Create a concat file listing the clips
        concat_file = os.path.join(final_video_dir, "concat_list.txt")
        with open(concat_file, 'w') as f:
            for clip_path in combined_clips:
                f.write(f"file '{os.path.abspath(clip_path)}'\n")
        
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

if __name__ == "__main__":
    print("Testing complete workflow with two countries...")
    if test_complete_workflow():
        print("\nComplete workflow test passed!")
    else:
        print("\nComplete workflow test failed!") 