# Women Around The World API Clients

This directory contains API clients for interacting with various services used in the Women Around The World project.

## Available Clients

- `TensorArtClient`: For image generation using TensorArt API
- `RunwayClient`: For animation generation using RunwayML API
- `ElevenLabsClient`: For text-to-speech using ElevenLabs API

## Usage Examples

### TensorArt Client

```python
from watw.api.clients.tensorart import TensorArtClient

# Initialize client
client = TensorArtClient()

# Generate an image
image_path, image_url = client.generate_image(
    prompt_text="A beautiful landscape with mountains and a lake",
    output_path="output/image.png"
)
```

### Runway Client

```python
from watw.api.clients.runway import RunwayClient

# Initialize client
client = RunwayClient()

# Generate an animation
animation_path = client.generate_animation(
    base_image_path="input/base_image.png",
    prompt="Create a smooth panning animation across the landscape",
    output_path="output/animation.mp4",
    seed=42
)
```

### ElevenLabs Client

```python
from watw.api.clients.elevenlabs import ElevenLabsClient

# Initialize client
client = ElevenLabsClient()

# Get available voices
voices = client.get_available_voices()

# Generate voiceover
speech_path = client.generate_voiceover(
    text="Hello, this is a test voiceover.",
    voice_id=voices[0]["voice_id"],
    output_path="output/voiceover.mp3"
)
```

## Error Handling

All clients raise appropriate exceptions when errors occur:

- `APIError`: Base exception for API-related errors
- `TensorArtError`: For TensorArt API errors
- `RunwayMLError`: For RunwayML API errors
- `ElevenLabsError`: For ElevenLabs API errors

Example error handling:

```python
from watw.utils.common.exceptions import APIError

try:
    # Use any client method
    result = client.some_method()
except APIError as e:
    print(f"API error occurred: {str(e)}")
```

## Configuration

API clients are configured using environment variables:

- `TENSORART_API_KEY`: TensorArt API key
- `RUNWAYML_API_KEY`: RunwayML API key
- `ELEVENLABS_API_KEY`: ElevenLabs API key

You can also pass these values directly when initializing the clients:

```python
client = TensorArtClient(api_key="your-api-key")
```

## Rate Limiting and Retries

All clients include built-in rate limiting and retry mechanisms:

- Rate limiting is configured per service
- Retries are handled automatically for transient failures
- Custom retry configurations can be provided

Example with custom retry configuration:

```python
from watw.utils.api.retry import RetryConfig

retry_config = RetryConfig(
    max_retries=5,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True
)

client = TensorArtClient(retry_config=retry_config)
```