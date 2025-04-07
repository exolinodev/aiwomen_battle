#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key, play

def test_elevenlabs_connection():
    """Test the connection to ElevenLabs API and generate a simple voice-over."""
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
        
        # Generate a simple test audio
        print("Generating test audio...")
        audio = generate(
            text="This is a test of the ElevenLabs voice-over integration.",
            voice="Bella",
            model="eleven_multilingual_v2"
        )
        
        # Save the audio to a file
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "test_voiceover.mp3")
        save(audio, output_path)
        print(f"Test audio saved to: {output_path}")
        
        # Play the audio
        print("Playing test audio...")
        play(audio)
        
        return True
    
    except Exception as e:
        print(f"Error testing ElevenLabs integration: {e}")
        return False

def test_voiceover_generation():
    """Test the voice-over generation with specific parameters."""
    try:
        # Import the voiceover module
        sys.path.append(os.path.dirname(__file__))
        from Generation.voiceover import generate_voiceover_for_video
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a voice-over
        print("Generating voice-over for test video...")
        voiceover_path = generate_voiceover_for_video(
            video_number=1,
            country1="Japan",
            country2="France",
            output_dir=output_dir
        )
        
        if voiceover_path and os.path.exists(voiceover_path):
            print(f"Voice-over generated successfully: {voiceover_path}")
            return True
        else:
            print("Failed to generate voice-over")
            return False
    
    except Exception as e:
        print(f"Error testing voice-over generation: {e}")
        return False

if __name__ == "__main__":
    print("Testing ElevenLabs integration...")
    if test_elevenlabs_connection():
        print("\nElevenLabs API connection test passed!")
    else:
        print("\nElevenLabs API connection test failed!")
    
    print("\nTesting voice-over generation...")
    if test_voiceover_generation():
        print("\nVoice-over generation test passed!")
    else:
        print("\nVoice-over generation test failed!") 