#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from datetime import datetime
import time

# Import from your refactored scripts
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from Generation.render import (
        check_credentials,
        generate_base_image_tensorart,
        generate_animation_runway,
        ANIMATION_PROMPTS,  # Import the prompts list
        ANIMATION_SEED_START # Import start seed
    )
    from experiment import EnhancedRhythmicVideoEditor
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure render.py and experiment.py are in the correct directories.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Generate animated clips and combine them into a rhythmic video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # --- Workflow Arguments ---
    parser.add_argument("--audio", "-a", type=str, required=True, help="Path to the audio file for the rhythmic video.")
    parser.add_argument("--output-base-dir", "-o", type=str, default="workflow_output", help="Base directory for all generated files and outputs.")
    parser.add_argument("--output-name", "-on", type=str, default=None, help="Final rhythmic video filename (no extension, timestamp added). Default: rhythmic_video_TIMESTAMP.mp4")
    parser.add_argument("--editor-config", "-ecfg", type=str, default=None, help="Path to a JSON config file for the EnhancedRhythmicVideoEditor.")
    parser.add_argument("--keep-temp", "-k", action="store_true", help="Keep all temporary files (animator base image, editor temp files) after completion.")

    args = parser.parse_args()
    start_time = time.time()

    # --- Setup Directory Structure ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = f"{args.output_base_dir}_{timestamp}" # Unique base dir per run

    # Directory for generated animation clips (becomes editor input)
    generated_clips_dir = os.path.join(base_dir, "generated_clips")
    # Directory for temporary files (animator base image, editor processing)
    temp_files_dir = os.path.join(base_dir, "temp_files")
    # Directory for final editor output video and config
    final_video_dir = os.path.join(base_dir, "final_video")

    try:
        os.makedirs(generated_clips_dir, exist_ok=True)
        os.makedirs(temp_files_dir, exist_ok=True)
        os.makedirs(final_video_dir, exist_ok=True)
        print(f"Created directory structure in: {base_dir}")
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
            # Decide if you want to continue or abort on failure
            # continue
            # sys.exit(1) # Abort example
        current_seed += 1

    if not generated_clips_paths:
        print("Workflow aborted: No animation clips were successfully generated.")
        # Clean up temp dir if desired even on failure
        if not args.keep_temp:
             try:
                 print(f"Cleaning up temporary directory: {temp_files_dir}")
                 shutil.rmtree(temp_files_dir)
             except Exception as e:
                 print(f"Warning: Error during cleanup: {e}")
        sys.exit(1)

    print(f"\nSuccessfully generated {len(generated_clips_paths)} animation clips in '{generated_clips_dir}'")

    # --- Stage 2: Rhythmic Video Editing ---
    print("\n" + "="*20 + " Stage 2: Creating Rhythmic Video " + "="*20)

    # Define final video output path
    output_video_name_base = args.output_name if args.output_name else f"rhythmic_video_{timestamp}"
    final_output_file = os.path.join(final_video_dir, f"{output_video_name_base}.mp4")
    # Define editor's specific temp directory *within* the main temp_files_dir
    editor_temp_dir = os.path.join(temp_files_dir, "editor_temp")
    os.makedirs(editor_temp_dir, exist_ok=True)

    editor = None # Initialize editor variable
    try:
        # Initialize editor - pass generated clips dir as input, use specific temp dir
        editor = EnhancedRhythmicVideoEditor(
            clips_dir=generated_clips_dir,
            audio_path=args.audio,
            output_file=final_output_file,
            temp_dir=editor_temp_dir, # Use the dedicated sub-directory
            config=None # Config loaded below if specified
        )

        # Load editor config file if provided
        if args.editor_config:
            print("-" * 40)
            if os.path.exists(args.editor_config):
                loaded_conf = editor.load_config_file(args.editor_config)
                if loaded_conf is None: print(f"Warning: Failed to load editor config '{args.editor_config}'. Using defaults.")
                else: print("Editor config file loaded.")
            else: print(f"Warning: Editor config file specified but not found: {args.editor_config}. Using defaults.")
            print("-" * 40)

        # Run the editor workflow
        print("\nStarting video editor processing...")
        print(f" Input Clips Dir : {editor.clips_dir}")
        print(f" Audio Source    : {os.path.basename(editor.audio_path)}")
        print(f" Output Target   : {editor.output_file}")
        print(f" Editor Temp Dir : {editor.temp_dir}")
        print("-" * 60)

        editor.find_video_clips()
        editor.detect_beats()
        final_video_path = editor.create_beat_synchronized_video()

        # Save the *actually used* editor configuration
        config_save_path = os.path.join(final_video_dir, f"{output_video_name_base}_config_used.json")
        editor.create_config_file(config_file=config_save_path)

        print("\n" + "=" * 60 + "\n Workflow Completed Successfully\n" + "=" * 60)
        print(f" Final video saved to: {final_video_path}")

    except Exception as e:
        import traceback
        print("\n" + "!" * 60 + "\n Error During Rhythmic Video Editing\n" + "!" * 60, file=sys.stderr)
        print(f" Error Type: {type(e).__name__}", file=sys.stderr)
        print(f" Error Message: {str(e)}", file=sys.stderr)
        print(" Traceback:", file=sys.stderr); traceback.print_exc(file=sys.stderr)
        # Decide on cleanup on failure
        if not args.keep_temp:
            print(f"\nAttempting cleanup despite error...")
        else:
            print(f"\nError occurred, temporary files kept in: {temp_files_dir}")
            print(f"Generated clips (if any) kept in: {generated_clips_dir}")
        sys.exit(1) # Exit with error code
    finally:
        # --- Cleanup ---
        elapsed_time = time.time() - start_time
        print(f"\nTotal workflow time: {elapsed_time:.2f} seconds.")

        if not args.keep_temp:
            print("\nCleaning up temporary files...")
            try:
                # Remove the entire main temp directory
                shutil.rmtree(temp_files_dir)
                print(f"Removed temporary directory: {temp_files_dir}")
                # Optionally remove the generated clips now they're in the video
                # shutil.rmtree(generated_clips_dir)
                # print(f"Removed generated clips directory: {generated_clips_dir}")
            except Exception as e:
                print(f"Warning: Error during cleanup: {e}")
        else:
            print(f"\nTemporary files kept in: {temp_files_dir}")
            print(f"Generated clips kept in: {generated_clips_dir}")

if __name__ == "__main__":
    # Basic check for ffmpeg/ffprobe (optional but good practice)
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: ffmpeg or ffprobe not found or not working. Please install FFmpeg.", file=sys.stderr)
        sys.exit(1)

    main() 