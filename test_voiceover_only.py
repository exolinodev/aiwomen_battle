#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, play

def test_voiceover_generation():
    """Test the voice-over generation with specific parameters."""
    try:
        # Load environment variables
        env_path = Path("Generation/.env")
        load_dotenv(dotenv_path=env_path)
        
        # Get API key
        api_key = os.getenv("ELEVENLABS_API_SECRET")
        if not api_key:
            print("Error: ELEVENLABS_API_SECRET not found in .env file")
            return False
        
        # Set API key
        set_api_key(api_key)
        print("Successfully connected to ElevenLabs API")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a voice-over for video #1
        print("Generating voice-over for Video #1...")
        video_number = 1
        country1 = "Japan"
        country2 = "France"
        
        # Create the script
        script = f"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. "
        script += "Don't forget to like and subscribe to our channel for more amazing content!"
        
        print(f"Script: {script}")
        
        # Generate audio
        print("Generating audio...")
        audio = generate(
            text=script,
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        output_path = os.path.join(output_dir, f"voiceover_{video_number}.mp3")
        save(audio, output_path)
        print(f"Voice-over saved to: {output_path}")
        
        # Play the audio
        print("Playing voice-over...")
        play(audio)
        
        return True
    
    except Exception as e:
        print(f"Error testing voice-over generation: {e}")
        return False

if __name__ == "__main__":
    print("Testing ElevenLabs voice-over generation...")
    if test_voiceover_generation():
        print("\nVoice-over generation test passed!")
    else:
        print("\nVoice-over generation test failed!") 