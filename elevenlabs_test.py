from dotenv import load_dotenv
import os
from elevenlabs import generate, play, set_api_key
from pathlib import Path

# Load environment variables from the Generation directory
env_path = Path("Generation/.env")
load_dotenv(dotenv_path=env_path)

# Set the API key
set_api_key(os.getenv("ELEVENLABS_API_SECRET"))

# Generate audio from text
audio = generate(
    text="What country do you like most?",
    voice="Bella",  # Using Bella's voice
    model="eleven_multilingual_v2"
)

# Play the generated audio
play(audio) 