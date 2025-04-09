#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from datetime import datetime
import logging
from typing import List, Dict, Optional, Any, Union, Tuple

# Import from your refactored scripts
try:
    sys.path.append(os.path.dirname(__file__))
    from src.watw.core.render import (
        check_credentials,
        generate_all_base_images,
        generate_animation_runway,
        ANIMATION_PROMPTS,  # Import the prompts list
        ANIMATION_SEED_START # Import start seed
    )
    from src.watw.core.voiceover import generate_voiceover_for_video
    from src.watw.core.video_editor import create_final_video
    from src.watw.utils.common.logging_utils import setup_logger
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py, voiceover.py, and video_editor.py are in the correct directory.")
    sys.exit(1)

# Set up logging
logger = setup_logger(
    name="workflow",
    level=logging.INFO,
    log_file="workflow.log"
)

def main() -> None:
    # Create timestamped output directory
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base_dir: str = f"workflow_output_{timestamp}"
    
    # Directory for generated animation clips (becomes editor input)
    generated_clips_dir: str = os.path.join(output_base_dir, "generated_clips")
    # Directory for temporary files (animator base images)
    temp_files_dir: str = os.path.join(output_base_dir, "temp_files")
    # Directory for final editor output video
    final_video_dir: str = os.path.join(output_base_dir, "final_video")
    # Directory for voice-over audio files
    voiceover_dir: str = os.path.join(output_base_dir, "voiceovers")
    
    try:
        os.makedirs(generated_clips_dir, exist_ok=True)
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(final_video_dir, exist_ok=True)
        os.makedirs(voiceover_dir, exist_ok=True)
        logger.info(f"Created directory structure in: {output_base_dir}")
    except OSError as e:
        logger.error(f"Error creating directories: {e}")
        sys.exit(1)
    
    # --- Stage 1: Animation Generation ---
    logger.info("="*20 + " Stage 1: Generating Animation Clips " + "="*20)
    if not check_credentials():
        sys.exit(1)
    
    # 1a. Generate All Base Images
    logger.info("Generating all base images...")
    base_images: List[Dict[str, Any]] = generate_all_base_images(output_directory=temp_files_dir)
    
    if not base_images:
        logger.error("Workflow aborted: No base images were successfully generated.")
        sys.exit(1)
    
    logger.info(f"Successfully generated {len(base_images)} base images in '{temp_files_dir}'")
    
    # 1b. Generate Animations for Each Base Image
    generated_clips_paths: List[str] = []
    current_seed: int = ANIMATION_SEED_START
    
    for base_image in base_images:
        base_image_id: str = base_image["id"]
        base_image_path: str = base_image["path"]
        
        logger.info(f"Generating animation for base image: {base_image_id}")
        
        # Use the first animation prompt for each base image
        anim_prompt: Dict[str, Any] = ANIMATION_PROMPTS[0]
        prompt_id: str = anim_prompt["id"]
        prompt_text: str = anim_prompt["text"]
        
        # Save animation clips directly to the dedicated generated_clips_dir
        output_filename_base: str = f"animation_{base_image_id}"
        video_path: Optional[str] = generate_animation_runway(
            base_image_path=base_image_path,
            animation_prompt_text=prompt_text,
            output_directory=generated_clips_dir, # Use dedicated dir
            output_filename_base=output_filename_base,
            seed=current_seed
        )
        
        if video_path:
            generated_clips_paths.append(video_path)
            logger.info(f"Animation for {base_image_id} saved to: {video_path}")
        else:
            logger.warning(f"Failed to generate animation for base image: {base_image_id}")
        
        current_seed += 1
    
    if not generated_clips_paths:
        logger.error("Workflow aborted: No animation clips were successfully generated.")
        sys.exit(1)
    
    logger.info(f"Successfully generated {len(generated_clips_paths)} animation clips in '{generated_clips_dir}'")
    
    # --- Stage 2: Voice-over Generation ---
    logger.info("="*20 + " Stage 2: Generating Voice-over " + "="*20)
    
    # Example: Generate voice-over for video #1 with countries from the first animation
    # In a real implementation, you would extract country names from your prompts or data
    video_number: int = 1
    country1: str = "Japan"  # Example country, replace with actual country from your data
    country2: str = "France"  # Example country, replace with actual country from your data
    
    logger.info(f"Generating voice-over for Video #{video_number}...")
    voiceover_path: Optional[str] = generate_voiceover_for_video(
        video_number=video_number,
        country1=country1,
        country2=country2,
        output_dir=voiceover_dir
    )
    logger.info(f"Voice-over generated and saved to: {voiceover_path}")
    
    # --- Stage 3: Video Editing ---
    logger.info("="*20 + " Stage 3: Video Editing " + "="*20)
    
    # Optional: Path to background music
    background_music_path: str = os.path.join("music", "background_music.mp3")  # Adjust path as needed
    
    # Create final video with voice-over
    logger.info("Creating final video with voice-over...")
    final_video_path: Optional[str] = create_final_video(
        video_clips_dir=generated_clips_dir,
        voiceover_path=voiceover_path,
        output_dir=final_video_dir,
        background_music_path=background_music_path if os.path.exists(background_music_path) else None,
        music_volume=0.1  # Adjust volume as needed
    )
    
    if final_video_path:
        logger.info(f"Final video created successfully: {final_video_path}")
    else:
        logger.error("Failed to create final video.")
    
    logger.info("Workflow completed successfully!")
    logger.info(f"Output directory: {output_base_dir}")
    logger.info(f"Generated clips: {generated_clips_dir}")
    logger.info(f"Voice-over: {voiceover_path}")
    logger.info(f"Final video: {final_video_path}")

if __name__ == "__main__":
    main() 