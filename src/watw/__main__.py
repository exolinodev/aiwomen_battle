#!/usr/bin/env python3
"""
Women Around The World - Main Entry Point
"""

# Standard library imports
import argparse
from pathlib import Path

# Local imports
from watw.utils.common.api_utils import APIConfiguration
from watw.core import VideoEditor, VoiceoverGenerator, VideoRenderer
from watw.utils import (
    ensure_directory,
    setup_logger,
    log_execution_time,
)

# Set up logger
logger = setup_logger(name="watw.main")

@log_execution_time(logger)
def main():
    parser = argparse.ArgumentParser(description="Women Around The World Video Generator")
    parser.add_argument("--country", required=True, help="Country code to generate video for")
    parser.add_argument("--output-dir", default="output", help="Output directory for generated content")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--log-file", help="Path to log file")
    
    args = parser.parse_args()
    
    # Set up file logging if specified
    if args.log_file:
        setup_logger(name="watw.main", log_file=args.log_file)
    
    # Load API configuration
    api_config = APIConfiguration(args.config)
    
    # Create output directory
    output_dir = ensure_directory(args.output_dir)
    
    # Initialize components with config
    voiceover_gen = VoiceoverGenerator(api_config)
    video_editor = VideoEditor(api_config)
    renderer = VideoRenderer(api_config)
    
    # Generate video
    final_video = renderer.create_video(args.country, output_dir)
    
    logger.info(f"Video generated successfully: {final_video}")

if __name__ == "__main__":
    main() 