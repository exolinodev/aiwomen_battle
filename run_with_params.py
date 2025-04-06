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
        generate_base_image_tensorart,
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
    # Directory for temporary files (animator base image)
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
    
    # 1a. Generate Base Image
    print("Generating base image...")
    # Save base image in the *temp* directory
    base_image_local_path, _ = generate_base_image_tensorart(output_directory=temp_files_dir)
    if not base_image_local_path:
        print("Workflow aborted: Base image generation failed.")
        sys.exit(1)
    print(f"Base image saved temporarily to: {base_image_local_path}")
    
    # 1b. Generate Animations
    generated_clips_paths = []
    current_seed = ANIMATION_SEED_START
    for anim_prompt in ANIMATION_PROMPTS:
        prompt_id = anim_prompt["id"]
        prompt_text = anim_prompt["text"]
        # Save animation clips directly to the dedicated generated_clips_dir
        output_filename_base = f"animation_{prompt_id}"
        print(f"\nGenerating animation for: {prompt_id}")
        video_path = generate_animation_runway(
            base_image_path=base_image_local_path,
            animation_prompt_text=prompt_text,
            output_directory=generated_clips_dir, # Use dedicated dir
            output_filename_base=output_filename_base,
            seed=current_seed
        )
        if video_path:
            generated_clips_paths.append(video_path)
        else:
            print(f"Warning: Failed to generate animation for prompt: {prompt_id}")
        current_seed += 1
    
    if not generated_clips_paths:
        print("Workflow aborted: No animation clips were successfully generated.")
        sys.exit(1)
    
    print(f"\nSuccessfully generated {len(generated_clips_paths)} animation clips in '{generated_clips_dir}'")
    
    # --- Stage 2: Rhythmic Video Editing ---
    print("\n" + "="*20 + " Stage 2: Creating Rhythmic Video " + "="*20)
    
    # Define final video output path
    output_video_name = f"rhythmic_video_{timestamp}"
    final_output_file = os.path.join(final_video_dir, f"{output_video_name}.mp4")
    
    # Build the command for experiment.py using parameters from coammnds.txt
    audio_file = "/Users/dev/womanareoundtheworld/Music_sync/Done/spain/trash/Pasión en el Aire.mp3"
    
    cmd = [
        "python", "experiment.py",
        "--input-dir", generated_clips_dir,
        "--audio", audio_file,
        "--duration", "30",
        "--beat-sensitivity", "0.3",
        "--min-segment", "2.0",
        "--max-segment", "6.0",
        "--min-speed", "0.7",
        "--max-speed", "1.1",
        "--speed-change-prob", "0.2",
        "--transition", "crossfade",
        "--transition-duration", "0.8",
        "--transition-safe",
        "--hard-cut-ratio", "0.1",
        "--rand-clip-strength", "0.95",
        "--video-fadeout-duration", "2",
        "--output-name", output_video_name,
        "--quality", "high",
        "--audio-fade-in", "1.0",
        "--audio-fade-out", "2.0",
        "--output-dir", final_video_dir,
        "--temp-dir", temp_files_dir
    ]
    
    print("\nExecuting experiment.py with parameters from coammnds.txt:")
    print(" ".join(cmd))
    
    try:
        # Run the experiment.py script
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("\n" + "=" * 60 + "\n Workflow Completed Successfully\n" + "=" * 60)
        print(f" Final video saved to: {final_output_file}")
        print("\nOutput from experiment.py:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\n" + "!" * 60 + "\n Error During Rhythmic Video Editing\n" + "!" * 60, file=sys.stderr)
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