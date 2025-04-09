# Women Around The World

A Python project for generating educational videos about women's experiences around the world.

## Project Structure

```
watw/
├── src/
│   └── watw/
│       ├── core/           # Core functionality
│       │   ├── video_editor.py
│       │   ├── voiceover.py
│       │   └── render.py
│       ├── utils/          # Utility functions
│       │   ├── common/     # Common utilities
│       │   │   ├── file_utils.py
│       │   │   ├── logging_utils.py
│       │   │   └── validation_utils.py
│       │   ├── countries.py
│       │   └── prompts.py
│       └── config/         # Configuration management
│           ├── __init__.py
│           └── config.py
├── tests/                  # Test files
├── config/                 # Configuration files
│   ├── config.json         # Your configuration file
│   ├── config.example.json # Example configuration
│   └── .env               # Environment variables
└── setup.py               # Package setup file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/watw.git
cd watw
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

## Configuration

The project uses a centralized configuration system that combines:

1. Default values in the code
2. Configuration file (`config/config.json`)
3. Environment variables (`.env` file)

### Setting Up Configuration

1. Copy the example configuration file:
```bash
cp config/config.example.json config/config.json
```

2. Edit `config/config.json` with your settings:
```json
{
    "elevenlabs_api_key": "your-elevenlabs-api-key-here",
    "runway_api_key": "your-runway-api-key-here",
    "video_width": 1920,
    "video_height": 1080,
    "video_fps": 30,
    "video_duration": 60,
    "audio_sample_rate": 44100,
    "audio_channels": 2,
    "voice_id": "default",
    "voice_stability": 0.5,
    "voice_similarity_boost": 0.75,
    "output_format": "mp4",
    "temp_dir": "temp"
}
```

3. Create a `.env` file for sensitive information:
```
ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
RUNWAY_API_KEY=your-runway-api-key-here
```

### Configuration Priority

1. Environment variables (highest priority)
2. Configuration file values
3. Default values in code (lowest priority)

## Usage

Generate a video for a specific country:

```bash
python -m watw --country US --output-dir output
```

### Command Line Options

- `--country`: Country code to generate video for (required)
- `--output-dir`: Output directory for generated content (default: "output")
- `--config`: Path to configuration file (default: "config/config.json")
- `--log-file`: Path to log file (optional)

## Utilities

The project provides a set of common utilities that can be used across the codebase:

### File Utilities

```python
from watw.utils import ensure_directory, get_file_extension, create_temp_file, copy_file

# Ensure a directory exists
output_dir = ensure_directory("output")

# Get file extension
extension = get_file_extension("video.mp4")  # Returns ".mp4"

# Create a temporary file
temp_file = create_temp_file("content", extension=".txt")

# Copy a file
copy_file("source.txt", "destination.txt")
```

### Logging Utilities

```python
from watw.utils import setup_logger, log_execution_time, log_function_call

# Set up a logger
logger = setup_logger(name="my_module", log_file="logs/my_module.log")

# Log execution time of a function
@log_execution_time(logger)
def my_function():
    # Function implementation
    pass

# Log function calls
@log_function_call(logger)
def another_function(arg1, arg2):
    # Function implementation
    pass
```

### Validation Utilities

```python
from watw.utils import (
    ValidationError,
    validate_file_exists,
    validate_directory_exists,
    validate_required_fields,
    validate_file_extension,
    validate_api_key,
)

# Validate that a file exists
file_path = validate_file_exists("input.txt")

# Validate that a directory exists
dir_path = validate_directory_exists("output")

# Validate required fields in a dictionary
validate_required_fields(data, ["name", "age"])

# Validate file extension
validate_file_extension("video.mp4", [".mp4", ".avi"])

# Validate API key
validate_api_key(api_key, "ServiceName")
```

## Development

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
python -m pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 