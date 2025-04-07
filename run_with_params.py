#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from datetime import datetime

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from Generation.render import (
        check_credentials,
        generate_all_base_images,
        generate_animation_runway,
        ANIMATION_PROMPTS,  # Import the prompts list
        ANIMATION_SEED_START # Import start seed
    )
    from Generation.voiceover import generate_voiceover_for_video
    from Generation.video_editor import create_final_video
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py, voiceover.py, and video_editor.py are in the correct directory.")
    sys.exit(1)

def main():
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir = f"workflow_output_{timestamp}"
    
    # Directory for generated animation clips (becomes editor input)
    generated_clips_dir = os.path.join(output_base_dir, "generated_clips")
    # Directory for temporary files (animator base images)
    temp_files_dir = os.path.join(output_base_dir, "temp_files")
    # Directory for final editor output video
    final_video_dir = os.path.join(output_base_dir, "final_video")
    # Directory for voice-over audio files
    voiceover_dir = os.path.join(output_base_dir, "voiceovers")
    
    try:
        os.makedirs(generated_clips_dir, exist_ok=True)
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(final_video_dir, exist_ok=True)
        os.makedirs(voiceover_dir, exist_ok=True)
        print(f"Created directory structure in: {output_base_dir}")
    except OSError as e:
        print(f"Error creating directories: {e}")
        sys.exit(1)
    
    # --- Stage 1: Animation Generation ---
    print("\n" + "="*20 + " Stage 1: Generating Animation Clips " + "="*20)
    if not check_credentials():
        sys.exit(1)
    
    # 1a. Generate All Base Images
    print("Generating all base images...")
    base_images = generate_all_base_images(output_directory=temp_files_dir)
    
    if not base_images:
        print("Workflow aborted: No base images were successfully generated.")
        sys.exit(1)
    
    print(f"\nSuccessfully generated {len(base_images)} base images in '{temp_files_dir}'")
    
    # 1b. Generate Animations for Each Base Image
    generated_clips_paths = []
    current_seed = ANIMATION_SEED_START
    
    for base_image in base_images:
        base_image_id = base_image["id"]
        base_image_path = base_image["path"]
        
        print(f"\nGenerating animation for base image: {base_image_id}")
        
        # Use the first animation prompt for each base image
        anim_prompt = ANIMATION_PROMPTS[0]
        prompt_id = anim_prompt["id"]
        prompt_text = anim_prompt["text"]
        
        # Save animation clips directly to the dedicated generated_clips_dir
        output_filename_base = f"animation_{base_image_id}"
        video_path = generate_animation_runway(
            base_image_path=base_image_path,
            animation_prompt_text=prompt_text,
            output_directory=generated_clips_dir, # Use dedicated dir
            output_filename_base=output_filename_base,
            seed=current_seed
        )
        
        if video_path:
            generated_clips_paths.append(video_path)
            print(f"Animation for {base_image_id} saved to: {video_path}")
        else:
            print(f"Warning: Failed to generate animation for base image: {base_image_id}")
        
        current_seed += 1
    
    if not generated_clips_paths:
        print("Workflow aborted: No animation clips were successfully generated.")
        sys.exit(1)
    
    print(f"\nSuccessfully generated {len(generated_clips_paths)} animation clips in '{generated_clips_dir}'")
    
    # --- Stage 2: Voice-over Generation ---
    print("\n" + "="*20 + " Stage 2: Generating Voice-over " + "="*20)
    
    # Example: Generate voice-over for video #1 with countries from the first animation
    # In a real implementation, you would extract country names from your prompts or data
    video_number = 1
    country1 = "Japan"  # Example country, replace with actual country from your data
    country2 = "France"  # Example country, replace with actual country from your data
    
    print(f"Generating voice-over for Video #{video_number}...")
    voiceover_path = generate_voiceover_for_video(
        video_number=video_number,
        country1=country1,
        country2=country2,
        output_dir=voiceover_dir
    )
    print(f"Voice-over generated and saved to: {voiceover_path}")
    
    # --- Stage 3: Video Editing ---
    print("\n" + "="*20 + " Stage 3: Video Editing " + "="*20)
    
    # Optional: Path to background music
    background_music_path = os.path.join("music", "background_music.mp3")  # Adjust path as needed
    
    # Create final video with voice-over
    print("Creating final video with voice-over...")
    final_video_path = create_final_video(
        video_clips_dir=generated_clips_dir,
        voiceover_path=voiceover_path,
        output_dir=final_video_dir,
        background_music_path=background_music_path if os.path.exists(background_music_path) else None,
        music_volume=0.1  # Adjust volume as needed
    )
    
    if final_video_path:
        print(f"Final video created successfully: {final_video_path}")
    else:
        print("Failed to create final video.")
    
    print("\nWorkflow completed successfully!")
    print(f"Output directory: {output_base_dir}")
    print(f"Generated clips: {generated_clips_dir}")
    print(f"Voice-over: {voiceover_path}")
    print(f"Final video: {final_video_path}")

if __name__ == "__main__":
    main() 