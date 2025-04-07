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
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py is in the correct directory.")
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
    
    try:
        os.makedirs(generated_clips_dir, exist_ok=True)
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(final_video_dir, exist_ok=True)
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
    
    # --- Stage 2: Simple Video Concatenation ---
    print("\n" + "="*20 + " Stage 2: Creating Final Video " + "="*20)
    
    # Define final video output path
    output_video_name = f"swimsuit_around_world_{timestamp}"
    final_output_file = os.path.join(final_video_dir, f"{output_video_name}.mp4")
    
    # Create a file list for FFmpeg concatenation
    file_list_path = os.path.join(temp_files_dir, "file_list.txt")
    with open(file_list_path, "w") as f:
        for clip_path in generated_clips_paths:
            f.write(f"file '{clip_path}'\n")
    
    # Build the FFmpeg command for simple concatenation
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        final_output_file
    ]
    
    print("\nExecuting FFmpeg for simple concatenation:")
    print(" ".join(cmd))
    
    try:
        # Run the FFmpeg command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n" + "=" * 60 + "\n Workflow Completed Successfully\n" + "=" * 60)
        print(f" Final video saved to: {final_output_file}")
        print("\nOutput from FFmpeg:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\n" + "!" * 60 + "\n Error During Video Concatenation\n" + "!" * 60, file=sys.stderr)
        print(f" Error Type: {type(e).__name__}", file=sys.stderr)
        print(f" Error Message: {str(e)}", file=sys.stderr)
        print(" Output:", e.stdout)
        print(" Error:", e.stderr)
        sys.exit(1)
    
    print(f"\nTemporary files kept in: {temp_files_dir}")
    print(f"Generated clips kept in: {generated_clips_dir}")
    print(f"Final video saved in: {final_video_dir}")

if __name__ == "__main__":
    main() 