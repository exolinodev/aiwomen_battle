#!/usr/bin/env python3
import os
import sys
import time
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
    from Generation.video_editor import combine_video_with_voiceover
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py and video_editor.py are in the correct directory.")
    sys.exit(1)

def test_complete_workflow():
    """Test the complete workflow with a single country."""
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
        print("\n" + "="*20 + " Stage 1: Generating Base Image " + "="*20)
        if not check_credentials():
            print("Error: API credentials not valid")
            return False
        
        # Generate base image for a single country
        country_name = "Japan"
        country_code = "01_japan"
        print(f"Generating base image for: {country_code}")
        
        # Define the athletic-themed prompt for base image generation
        base_image_prompt = (
            "Generate a high-detail, cinematic static image of a female athlete captured mid-action "
            "during an intense moment in an international sports competition. Focus on a medium shot "
            "or close-up that highlights her expression of fierce concentration and determination. "
            "She wears generic, functional athletic team sportswear primarily in Red and White to "
            "subtly represent her nation/team. Ensure good lighting that emphasizes her athletic form "
            "and the intensity of the moment. The background should be relevant to the sport but softly "
            "blurred to keep focus on the athlete."
        )
        
        # Generate base image
        base_image_result = generate_base_image_tensorart(
            output_directory=temp_files_dir,
            prompt_text=base_image_prompt,
            prompt_id=None
        )
        
        if not base_image_result or not base_image_result[0] or not os.path.exists(base_image_result[0]):
            print(f"Error: Failed to generate base image for {country_code}")
            return False
        
        base_image_path = base_image_result[0]
        print(f"Base image for {country_code} saved to: {base_image_path}")
        
        # --- Stage 2: Animation Generation ---
        print("\n" + "="*20 + " Stage 2: Generating Animation " + "="*20)
        
        # Define the animation prompt
        animation_prompt = (
            "Animate the provided base image. Add subtle, realistic motion focused on the athlete's "
            "contained energy and the environment. Her hair should have a slight natural sway from "
            "movement or a faint breeze. Show very subtle muscle tension shifts or breathing motion "
            "consistent with her focused, ready-state action. If clothing is loose, add slight, natural "
            "ripples. The background elements (if any are distinct in the blur) can have minimal "
            "atmospheric movement (e.g., heat haze, slight dust/sand drift). Crucially, maintain her "
            "intense facial expression of concentration. Keep camera movement minimal or entirely static "
            "to emphasize the athlete's contained power just before or during the action."
        )
        
        # Generate animation
        animation_path = generate_animation_runway(
            base_image_path=base_image_path,
            animation_prompt_text=animation_prompt,
            output_directory=generated_clips_dir,
            output_filename_base=f"animation_{country_code}",
            seed=12345
        )
        
        if not animation_path or not os.path.exists(animation_path):
            print(f"Error: Failed to generate animation for {country_code}")
            return False
        
        print(f"Animation for {country_code} saved to: {animation_path}")
        
        # --- Stage 3: Voice-over Generation ---
        print("\n" + "="*20 + " Stage 3: Generating Voice-over " + "="*20)
        
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
        
        # Generate a voice-over for video #1
        print("Generating voice-over for Video #1...")
        video_number = 1
        
        # Create the script
        script = f"Video #{video_number}. What country do you like most? {country_name} from the image generation. "
        script += "Don't forget to like and subscribe to our channel for more amazing content!"
        
        print(f"Script: {script}")
        
        # Generate audio
        print("Generating audio...")
        audio = generate(
            text=script,
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        voiceover_path = os.path.join(voiceover_dir, f"voiceover_{video_number}.mp3")
        save(audio, voiceover_path)
        print(f"Voice-over saved to: {voiceover_path}")
        
        # --- Stage 4: Video Editing ---
        print("\n" + "="*20 + " Stage 4: Creating Final Video " + "="*20)
        
        # Output path for the final video
        final_video_path = os.path.join(final_video_dir, "final_video.mp4")
        
        # Combine video with voice-over
        print(f"Combining video with voice-over...")
        success = combine_video_with_voiceover(
            video_path=animation_path,
            voiceover_path=voiceover_path,
            output_path=final_video_path
        )
        
        if success and os.path.exists(final_video_path):
            print(f"Successfully created final video: {final_video_path}")
            return True
        else:
            print("Failed to create final video")
            return False
    
    except Exception as e:
        print(f"Error in complete workflow test: {e}")
        return False

if __name__ == "__main__":
    print("Testing complete workflow with a single country...")
    if test_complete_workflow():
        print("\nComplete workflow test passed!")
    else:
        print("\nComplete workflow test failed!") 