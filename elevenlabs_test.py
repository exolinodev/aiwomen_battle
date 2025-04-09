from dotenv import load_dotenv
import os
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from pathlib import Path

# Load environment variables from the Generation directory
env_path = Path("Generation/.env")
load_dotenv(dotenv_path=env_path)

# Initialize ElevenLabs client
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_SECRET"))

try:
    # List available voices
    voices = client.voices.get_all()
    print("Available voices:")
    for voice in voices:
        print(f"- {voice}")

    # Use the first available voice
    if voices:
        voice_id = voices[0]
        print(f"\nUsing voice ID: {voice_id}")

        # Generate audio from text
        audio = client.text_to_speech.convert(
            text="What country do you like most?",
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )

        # Play the generated audio
        play(audio)
    else:
        print("No voices available in your ElevenLabs account.")
except Exception as e:
    print(f"Error: {str(e)}") 