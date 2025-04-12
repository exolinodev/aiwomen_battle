# Women Around The World

A Python project for generating educational videos about women's experiences around the world.

## Project Structure

```
watw/
├── src/
│   └── watw/
│       ├── api/                # API clients
│       │   ├── clients/        # API client implementations
│       │   │   ├── base.py     # Base API client
│       │   │   ├── elevenlabs.py
│       │   │   ├── runway.py
│       │   │   └── tensorart.py
│       │   ├── config.py       # API configuration
│       │   └── utils/         
│       ├── core/               # Core functionality
│       │   ├── Generation/     # Content generation
│       │   ├── video/          # Video editing
│       │   │   ├── base.py     # Base VideoEditor class
│       │   │   ├── rhythmic.py 
│       │   │   └── enhanced.py 
│       │   ├── workflow/       # Workflow management
│       │   ├── voiceover.py    # Voice generation
│       │   └── render.py       # Rendering functionality
│       ├── utils/              # Utility functions
│       │   ├── api/            # API utilities
│       │   ├── common/         # Common utilities
│       │   │   ├── file_utils.py
│       │   │   ├── logging_utils.py
│       │   │   └── validation_utils.py
│       │   └── video/          # Video utilities
│       └── config/             # Configuration management
├── tests/                      # Test files
├── config/                     # Configuration files
│   ├── config.json             # Your configuration file
│   └── config.example.json     # Example configuration
└── setup.py                    # Package setup file
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
# For production use:
pip install -e .

# For development (includes testing and linting tools):
pip install -e ".[dev]"
```

## Development Tools

The project includes several development tools for testing, linting, and code formatting:

- **Testing**: pytest with coverage reporting
- **Type Checking**: mypy
- **Linting**: ruff (replaces flake8 and isort)
- **Code Formatting**: black
- **Mocking**: unittest-mock

To run the development tools:

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src

# Type checking
mypy src/

# Linting
ruff check src/

# Format code
black src/
ruff format src/
```

## Configuration

The project uses a centralized configuration system that combines:

1. Default values in the code
2. Configuration file (`config/config.json`)
3. Environment variables (using the `WATW_` prefix)

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

3. Alternatively, set environment variables with the `WATW_` prefix:
```
WATW_ELEVENLABS_API_KEY=your-elevenlabs-api-key-here
WATW_RUNWAY_API_KEY=your-runway-api-key-here
```

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

## API Clients

The project provides unified API clients for various services with robust error handling:

```python
from watw.api import elevenlabs, runway, tensorart
from watw.core.voiceover import VoiceoverGenerator, VoiceoverError

try:
    # Initialize the voiceover generator
    generator = VoiceoverGenerator()
    
    # Generate a voiceover with error handling
    audio_path = generator.generate_voiceover(
        video_number=1,
        country1="Japan",
        country2="France",
        output_path="output/voiceover.mp3",
        voice="Rachel",
        model="eleven_monolingual_v1"
    )
    print(f"Generated voice-over at: {audio_path}")
except VoiceoverError as e:
    print(f"Failed to generate voice-over: {e}")
    # Handle the error appropriately
```

### Error Handling

The project implements comprehensive error handling for all API operations:

1. **API Initialization Errors**
   - Missing or invalid API keys
   - Network connectivity issues
   - Configuration problems

2. **API Communication Errors**
   - Rate limiting
   - Invalid requests
   - Server errors

3. **File System Errors**
   - Permission issues
   - Disk space problems
   - Invalid file paths

4. **Unexpected Errors**
   - All other errors are caught and wrapped in appropriate exception types

Example error handling:
```python
try:
    generator = VoiceoverGenerator()
    output_path = generator.generate_voiceover(...)
except VoiceoverError as e:
    # Handle voice-over specific errors
    print(f"Voice-over generation failed: {e}")
except Exception as e:
    # Handle unexpected errors
    print(f"Unexpected error: {e}")
```

## Video Editing

The project includes a modular video editing system:

```python
from watw.core.video import VideoEditor, RhythmicVideoEditor, EnhancedVideoEditor

# Basic video editing
editor = VideoEditor()
editor.combine_video_with_audio(
    video_path="input/video.mp4",
    audio_path="input/audio.mp3",
    output_path="output/video_with_audio.mp4"
)

# Rhythmic video editing (synchronized with music)
rhythmic_editor = RhythmicVideoEditor()
rhythmic_editor.create_rhythmic_video(
    video_clips_dir="input/clips",
    audio_path="input/music.mp3",
    output_path="output/rhythmic_video.mp4"
)
```

## Testing

The project includes comprehensive test coverage with a focus on error handling:

1. **Unit Tests**
   - Test individual components in isolation
   - Mock external dependencies
   - Verify error handling

2. **Integration Tests**
   - Test component interactions
   - Verify API communication
   - Test file system operations

3. **Error Scenario Tests**
   - Test API initialization failures
   - Test API communication errors
   - Test file system errors
   - Test unexpected errors

Example test structure:
```python
from unittest.mock import patch, MagicMock
from watw.core.voiceover import VoiceoverGenerator, VoiceoverError
from watw.api.clients.base import APIError

class TestVoiceover(unittest.TestCase):
    def setUp(self):
        self.mock_client = MagicMock()
        self.patcher = patch('watw.core.voiceover.ElevenLabsClient', 
                           return_value=self.mock_client)
        self.patcher.start()
        
    def test_generate_voiceover(self):
        # Test successful voiceover generation
        generator = VoiceoverGenerator()
        output_path = generator.generate_voiceover(...)
        self.assertIsInstance(output_path, Path)
        
    def test_api_error(self):
        # Test API error handling
        self.mock_client.generate_and_save.side_effect = APIError("API error")
        generator = VoiceoverGenerator()
        with self.assertRaises(VoiceoverError):
            generator.generate_voiceover(...)
            
    def tearDown(self):
        self.patcher.stop()
```

### Running Tests

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Run tests:
```bash
python -m pytest tests/
```

3. Run tests with coverage:
```bash
python -m pytest --cov=src tests/
```

4. Type checking:
```bash
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.