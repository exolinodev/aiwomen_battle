"""
Example usage of the Women Around The World API clients.

This module demonstrates how to use the various API clients
for image generation, animation, and voiceover.
"""

from pathlib import Path
from typing import Optional, Tuple

from watw.api.clients.elevenlabs import ElevenLabsClient
from watw.api.clients.runway import RunwayClient
from watw.api.clients.tensorart import TensorArtClient
from watw.utils.common.exceptions import APIError

# Initialize clients
tensorart_client = TensorArtClient()
elevenlabs_client = ElevenLabsClient()
runway_client = RunwayClient()


def generate_voiceover_example(text: str, output_path: str) -> Optional[Path]:
    """
    Generate a voiceover using ElevenLabs.

    Args:
        text: Text to convert to speech
        output_path: Path to save the audio file

    Returns:
        Optional[Path]: Path to the generated audio file, or None if generation fails

    Raises:
        APIError: If voiceover generation fails
    """
    try:
        # Get available voices
        voices = elevenlabs_client.get_available_voices()
        if not voices:
            raise APIError("No voices available")

        # Use the first voice's name
        first_voice_name = voices[0].get("name") if voices else None
        if first_voice_name is None:
            raise APIError("Could not determine name for the first voice.")

        # Generate voiceover using generate_and_save
        speech_path = elevenlabs_client.generate_and_save(
            text=text,
            output_path=output_path,
            voice_name=first_voice_name,
            model_id="eleven_multilingual_v2",
        )

        return speech_path

    except Exception as e:
        raise APIError(f"Failed to generate voiceover: {str(e)}")


def generate_image_example(prompt: str, output_path: str) -> Optional[Tuple[Path, str]]:
    """
    Generate an image using TensorArt.

    Args:
        prompt: Image generation prompt
        output_path: Path to save the image

    Returns:
        Optional[Tuple[Path, str]]: Path to the generated image and image URL, or None if generation fails

    Raises:
        APIError: If image generation fails
    """
    try:
        # Generate image
        image_path, image_url = tensorart_client.generate_image(
            prompt_text=prompt,
            output_path=output_path,
            negative_prompt=None,
            width=None,
            height=None,
            steps=None,
            cfg_scale=None,
            sampler=None,
            model_id=None,
            seed=-1,
            timeout=300,
            polling_interval=10,
        )

        return image_path, image_url

    except Exception as e:
        raise APIError(f"Failed to generate image: {str(e)}")


def generate_animation_example(
    base_image_path: str, prompt: str, output_path: str, seed: int = 42
) -> Optional[Path]:
    """
    Generate an animation using RunwayML.

    Args:
        base_image_path: Path to the base image
        prompt: Animation prompt
        output_path: Path to save the animation
        seed: Random seed for generation

    Returns:
        Optional[Path]: Path to the generated animation, or None if generation fails

    Raises:
        APIError: If animation generation fails
    """
    try:
        # Generate animation
        animation_path = runway_client.generate_animation(
            image_path=base_image_path,
            prompt_text=prompt,
            output_path=output_path,
            seed=seed,
            model="gen-2",
        )

        return animation_path

    except Exception as e:
        raise APIError(f"Failed to generate animation: {str(e)}")


def main() -> int:
    """Run example usage of the API clients."""
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Generate voiceover
        print("\n--- Generating Voiceover ---")
        voiceover_path = generate_voiceover_example(
            "Hello, this is a test voiceover.", str(output_dir / "voiceover.mp3")
        )
        if voiceover_path:
            print(f"Generated voiceover: {voiceover_path}")
        else:
            print("Voiceover generation failed")

        # Generate base image
        print("\n--- Generating Base Image ---")
        image_result = generate_image_example(
            "A beautiful landscape with mountains and a lake",
            str(output_dir / "base_image.png"),
        )
        if image_result is not None:
            base_image_path, image_url = image_result
            print(f"Generated base image: {base_image_path}")
            print(f"Image URL: {image_url}")

            # Generate animation only if base image exists
            print("\n--- Generating Animation ---")
            animation_path = generate_animation_example(
                str(base_image_path),
                "Create a smooth panning animation across the landscape",
                str(output_dir / "animation.mp4"),
            )
            if animation_path:
                print(f"Generated animation: {animation_path}")
            else:
                print("Animation generation failed")
        else:
            print("Base image generation failed, skipping animation")

    except APIError as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
