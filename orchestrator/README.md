# Video Generation and Editing Workflow

This orchestrator script integrates two main components:
1. Animation Generation (using TensorArt and RunwayML)
2. Rhythmic Video Editing (using FFmpeg)

## Prerequisites

- Python 3.6+
- FFmpeg and FFprobe installed
- Required Python packages:
  - numpy
  - librosa
  - scipy
  - requests
  - python-dotenv
  - runwayml

## Environment Setup

1. Create a `.env` file in the project root with your API credentials:
```
TENSORART_BEARER_TOKEN=your_tensorart_token
RUNWAYML_API_SECRET=your_runwayml_secret
```

## Usage

Run the orchestrator script with required arguments:

```bash
python main_workflow.py --audio /path/to/your/music.mp3 --output-base-dir ./my_video_project
```

### Optional Arguments

- `--output-name, -on`: Custom name for the final video (default: rhythmic_video_TIMESTAMP.mp4)
- `--editor-config, -ecfg`: Path to a JSON config file for the video editor
- `--keep-temp, -k`: Keep temporary files after completion (useful for debugging)

### Example with All Options

```bash
python main_workflow.py \
  --audio /path/to/music.mp3 \
  --output-base-dir ./my_video_project \
  --output-name my_cool_video \
  --editor-config ./editor_config.json \
  --keep-temp
```

## Output Structure

The script creates a timestamped directory with the following structure:

```
workflow_output_YYYYMMDD_HHMMSS/
├── generated_clips/     # Animation clips from RunwayML
├── temp_files/         # Temporary files (base image, editor temp)
│   └── editor_temp/    # Editor-specific temporary files
└── final_video/        # Final output video and config
    ├── my_cool_video.mp4
    └── my_cool_video_config_used.json
```

## Error Handling

- The script includes comprehensive error handling and logging
- Temporary files are preserved if `--keep-temp` is used
- Each stage (animation generation, video editing) can be debugged independently

## Configuration

### Editor Configuration

Create a JSON file with editor settings:

```json
{
  "transition_types": ["crossfade", "fade"],
  "visual_effects": ["none", "vignette"],
  "speed_effects": ["normal", "slow_motion"],
  "quality_level": "high"
}
```

### Animation Prompts

Animation prompts are defined in `Generation/render.py`. Modify `ANIMATION_PROMPTS` to change the animation sequence.

## Troubleshooting

1. If FFmpeg is not found:
   ```bash
   brew install ffmpeg  # macOS
   # or
   apt-get install ffmpeg  # Ubuntu/Debian
   ```

2. If API credentials are not working:
   - Verify your `.env` file exists and contains correct credentials
   - Check if the APIs are accessible from your network

3. If temporary files are not cleaned up:
   - Use `--keep-temp` to inspect the files
   - Check file permissions in the output directory 