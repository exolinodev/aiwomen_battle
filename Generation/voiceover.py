import os
from pathlib import Path
from dotenv import load_dotenv
from elevenlabs import generate, save, set_api_key

# Load environment variables
env_path = Path("Generation/.env")
load_dotenv(dotenv_path=env_path)

# Set the API key
set_api_key(os.getenv("ELEVENLABS_API_SECRET"))

def generate_voiceover(
    video_number,
    country1,
    country2,
    output_path,
    voice="Bella",
    model="eleven_multilingual_v2"
):
    """
    Generate a voice-over for a video with the specified parameters.
    
    Args:
        video_number (int): The video number (1-100)
        country1 (str): The first country name
        country2 (str): The second country name
        output_path (str): Path to save the audio file
        voice (str): The voice to use (default: "Bella")
        model (str): The model to use (default: "eleven_multilingual_v2")
        
    Returns:
        str: Path to the generated audio file
    """
    # Create the script
    script = f"Video #{video_number}. What country do you like most? {country1} from the image generation, then {country2}. "
    script += "Don't forget to like and subscribe to our channel for more amazing content!"
    
    # Generate audio
    audio = generate(
        text=script,
        voice=voice,
        model=model
    )
    
    # Save the audio to a file
    save(audio, output_path)
    
    return output_path

def generate_voiceover_for_video(video_number, country1, country2, output_dir):
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
    output_path = os.path.join(output_dir, f"voiceover_{video_number}.mp3")
    
    # Generate voice-over
    return generate_voiceover(video_number, country1, country2, output_path) 