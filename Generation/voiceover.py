import os
import json
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import save

# Load environment variables
env_path: Path = Path("Generation/.env")
load_dotenv(dotenv_path=env_path)

def load_config():
    """Load configuration from config.json"""
    config_path = Path("config.json")
    with open(config_path) as f:
        return json.load(f)

def get_voice_id(client: ElevenLabs, voice_name: str = "Rachel") -> str:
    """
    Get the voice ID for a given voice name.
    
    Args:
        client (ElevenLabs): The ElevenLabs client
        voice_name (str): The name of the voice to use (default: "Rachel")
        
    Returns:
        str: The voice ID
    """
    voices = client.voices.get_all()
    for voice in voices.voices:
        if voice.name.lower() == voice_name.lower():
            return voice.voice_id
    # If voice not found, return the first available voice
    return voices.voices[0].voice_id

def generate_voiceover(
    video_number: int,
    country1: str,
    country2: str,
    output_path: str,
    voice: str = "Rachel",
    model: str = "eleven_multilingual_v2"
) -> str:
    """
    Generate a voice-over for a video with the specified parameters.
    
    Args:
        video_number (int): The video number (1-100)
        country1 (str): The first country name
        country2 (str): The second country name
        output_path (str): Path to save the audio file
        voice (str): The voice name to use (default: "Rachel")
        model (str): The model to use (default: "eleven_multilingual_v2")
        
    Returns:
        str: Path to the generated audio file
    """
    # Load config
    config = load_config()
    api_key = config.get("elevenlabs", {}).get("api_key")
    if not api_key:
        raise ValueError("elevenlabs.api_key not found in config.json")

    # Create the script
    script: str = f"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. "
    script += "Don't forget to like and subscribe to our channel for more amazing content!"
    
    # Initialize client
    client = ElevenLabs(api_key=api_key)
    
    # Get voice ID
    voice_id = get_voice_id(client, voice)
    
    # Generate audio
    audio = client.text_to_speech.convert(
        text=script,
        voice_id=voice_id,
        model_id=model,
        output_format="mp3_44100_128"
    )
    
    # Save the audio to a file
    save(audio, output_path)
    
    return output_path

def generate_voiceover_for_video(
    video_number: int, 
    country1: str, 
    country2: str, 
    output_dir: str
) -> str:
    """
    Generate a voice-over for a video and save it to the specified directory.
    
    Args:
        video_number (int): The video number (1-100)
        country1 (str): The first country name
        country2 (str): The second country name
        output_dir (str): Directory to save the audio file
        
    Returns:
        str: Path to the generated audio file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output path
    output_path: str = os.path.join(output_dir, f"voiceover_{video_number}.mp3")
    
    # Generate voice-over
    return generate_voiceover(video_number, country1, country2, output_path) 