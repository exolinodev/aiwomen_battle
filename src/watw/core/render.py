"""
Image and animation generation module.

This module provides functions for generating base images using TensorArt API
and creating animations using RunwayML API. It includes utilities for API
interaction, file operations, and workflow management.
"""

import base64
import logging
import random
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, Literal, TypedDict

from dotenv import load_dotenv

from watw.api.clients.runway import RunwayClient
from watw.api.clients.tensorart import TensorArtClient
from watw.core.Generation.countries import CountryManager, BASE_IMAGE_NEGATIVE_PROMPT
from watw.utils.api.retry import RetryConfig, with_retry
from watw.utils.common.exceptions import (
    ConfigurationError,
    FileOperationError,
    RunwayMLError,
    TensorArtError,
    ValidationError,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---
ANIMATION_SEED_START = 42  # Starting seed for animation generation


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class VideoRenderer:
    """A class for rendering videos with various effects and transitions."""

    def __init__(self) -> None:
        """Initialize the VideoRenderer."""
        pass

    def render(
        self, input_path: Union[str, Path], output_path: Union[str, Path]
    ) -> Path:
        """Render a video with effects and transitions.

        Args:
            input_path: Path to the input video file
            output_path: Path where the rendered video should be saved

        Returns:
            Path to the rendered video file
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        return output_path


# Load environment variables
load_dotenv()

# Initialize API clients and country manager
tensorart_client = TensorArtClient(require_api_key=False)
runway_client = RunwayClient(require_api_key=False)
country_manager = CountryManager()

# Base Image Generation Parameters (TensorArt)
BASE_IMAGE_CONFIG = {
    "width": 768,
    "height": 1280,
    "steps": 30,
    "cfg_scale": 7,
    "sampler": "Euler",
    "model_id": tensorart_client.model_id,
    "seed": -1,
    "count": 1,
}

# Animation Parameters (RunwayML)
class AnimationConfig(TypedDict):
    model: Literal["gen3a_turbo"]
    duration: Literal[5, 10]
    ratio: Literal["768:1280", "1280:768"]
    seed_start: int

ANIMATION_CONFIG: AnimationConfig = {
    "model": "gen3a_turbo",
    "duration": 5,
    "ratio": "768:1280",
    "seed_start": ANIMATION_SEED_START,
}

# Retry Configurations
TENSORART_RETRY_CONFIG = RetryConfig(
    max_retries=5, base_delay=30.0, max_delay=300.0, jitter=True
)

RUNWAYML_RETRY_CONFIG = RetryConfig(
    max_retries=5, base_delay=60.0, max_delay=300.0, jitter=True
)

# Animation prompts for different types of videos
ANIMATION_PROMPTS = [
    {
        "id": "dynamic_motion",
        "text": "Create a dynamic and cinematic animation of an athlete in motion. The movement should be smooth and fluid, with a gentle camera motion that follows the athlete's graceful movements. The lighting should be dramatic and cinematic, enhancing the sense of motion and energy.",
    },
    {
        "id": "pose_transition",
        "text": "Create a smooth transition between athletic poses, with elegant and graceful movement. The camera should slowly rotate around the subject, capturing the beauty of the motion. Use dramatic lighting to enhance the visual impact.",
    },
    {
        "id": "dance_flow",
        "text": "Create a flowing dance-like animation with smooth, continuous movement. The camera should follow the motion in a cinematic way, with dynamic lighting that emphasizes the grace and fluidity of the movement.",
    },
]

# --- Credential Management ---


def check_credentials() -> bool:
    """
    Check if required API credentials are available.

    Returns:
        bool: True if credentials are available

    Raises:
        ConfigurationError: If required credentials are missing
    """
    try:
        # Get API keys from config
        tensorart_api_key = tensorart_client.config.get_api_key("tensorart")
        runway_api_key = runway_client.config.get_api_key("runwayml")
        
        if not tensorart_api_key or not runway_api_key:
            raise ConfigurationError("API credentials missing")
            
        # Set API keys
        tensorart_client.api_key = tensorart_api_key
        runway_client.api_key = runway_api_key
        
        logger.info("API credentials loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Credential error: {str(e)}")
        raise ConfigurationError(f"Failed to load credentials: {str(e)}")


# --- File Operations ---


@with_retry(config=RetryConfig(max_retries=3, base_delay=5.0))
def download_file(url: str, save_path: Union[str, Path]) -> Path:
    """
    Download a file from a URL.

    Args:
        url: Source URL
        save_path: Destination path

    Returns:
        Path: Path to downloaded file

    Raises:
        FileOperationError: If download fails
    """
    save_path = Path(save_path)

    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading from {url} to {save_path}...")

        response = tensorart_client.get(url, stream=True, timeout=180)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Successfully downloaded {save_path}")
        return save_path

    except Exception as e:
        raise FileOperationError(f"Failed to download file from {url}: {str(e)}")


def encode_image_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image file to base64.

    Args:
        image_path: Path to image file

    Returns:
        str: Base64 encoded image

    Raises:
        FileOperationError: If encoding fails
    """
    try:
        import io

        from PIL import Image

        with Image.open(image_path) as img:
            # Validate and resize if needed
            width, height = img.size
            if (
                width != BASE_IMAGE_CONFIG["width"]
                or height != BASE_IMAGE_CONFIG["height"]
            ):
                logger.warning(
                    f"Image dimensions {width}x{height} do not match required {BASE_IMAGE_CONFIG['width']}x{BASE_IMAGE_CONFIG['height']}"
                )
                img = img.resize(
                    (BASE_IMAGE_CONFIG["width"], BASE_IMAGE_CONFIG["height"]),
                    Image.Resampling.LANCZOS,
                )
                logger.info("Image has been resized")

            if img.mode != "RGB":
                img = img.convert("RGB")

            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr_bytes = img_byte_arr.getvalue()

            return base64.b64encode(img_byte_arr_bytes).decode("utf-8")

    except Exception as e:
        raise FileOperationError(f"Failed to encode image {image_path}: {str(e)}")


# --- TensorArt API Functions ---


def get_tensorart_job_status_endpoint(job_id: str) -> str:
    """Get the endpoint URL for checking TensorArt job status."""
    return f"{tensorart_client.base_url}/v1/jobs/{job_id}"


@with_retry(config=TENSORART_RETRY_CONFIG)
def submit_tensorart_job(headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """
    Submit a job to TensorArt API.

    Args:
        headers: Request headers
        payload: Job payload

    Returns:
        str: Job ID

    Raises:
        TensorArtError: If job submission fails
    """
    try:
        response = tensorart_client.create_job(
            prompt=payload["stages"][1]["diffusion"]["prompts"][0]["text"],
            negative_prompt=payload["stages"][1]["diffusion"]["prompts"][1]["text"],
            width=payload["stages"][1]["diffusion"]["width"],
            height=payload["stages"][1]["diffusion"]["height"],
            steps=payload["stages"][1]["diffusion"]["steps"],
            cfg_scale=payload["stages"][1]["diffusion"]["cfg_scale"],
            sampler=payload["stages"][1]["diffusion"]["sampler"],
            seed=payload["stages"][0]["inputInitialize"]["seed"],
        )
        return response
    except Exception as e:
        raise TensorArtError(f"Failed to submit TensorArt job: {str(e)}")


@with_retry(config=RetryConfig(max_retries=10, base_delay=10.0))
def poll_tensorart_job(
    headers: Dict[str, str], job_id: str, output_directory: Union[str, Path]
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Poll TensorArt job status until completion.

    Args:
        headers: Request headers
        job_id: Job ID
        output_directory: Directory to save output

    Returns:
        Tuple[Optional[Path], Optional[str]]: Path to saved image and image URL

    Raises:
        TensorArtError: If polling fails
    """
    try:
        job_status = tensorart_client.wait_for_job_completion(job_id)

        if job_status["status"] == "SUCCEEDED":
            image_url = tensorart_client.extract_image_url(job_status)
            if not image_url:
                raise TensorArtError("No image URL found in job status")

            # Download image
            output_path = Path(output_directory) / f"tensorart_{job_id}.png"
            download_file(image_url, output_path)

            return output_path, image_url
        else:
            raise TensorArtError(f"Job failed with status: {job_status['status']}")

    except Exception as e:
        raise TensorArtError(f"Failed to poll TensorArt job: {str(e)}")


def generate_base_image_tensorart(
    output_directory: Union[str, Path],
    prompt_id: Optional[str] = None,
    prompt_text: Optional[str] = None,
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Generate a base image using TensorArt.

    Args:
        output_directory: Directory to save output
        prompt_id: ID of prompt to use
        prompt_text: Custom prompt text

    Returns:
        Tuple[Optional[Path], Optional[str]]: Path to saved image and image URL

    Raises:
        TensorArtError: If generation fails
    """
    logger.info("\n--- Starting Base Image Generation (TensorArt) ---")

    if not tensorart_client.api_key:
        raise ConfigurationError("TensorArt API token missing")

    try:
        # Get prompt
        if prompt_id:
            country = country_manager.get_country(prompt_id)
            if not country:
                raise ValidationError(f"Invalid or missing country data for ID: {prompt_id}")
            prompt_text = country.base_image_prompt
            if not prompt_text:
                raise ValidationError(f"Missing 'base_image_prompt' for country ID: {prompt_id}")
        elif not prompt_text:
            raise ValidationError("Either prompt_id or prompt_text must be provided")

        # Create request ID
        request_id = str(uuid.uuid4())

        # Create request payload
        payload = {
            "request_id": request_id,
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {
                        "seed": BASE_IMAGE_CONFIG["seed"],
                        "count": BASE_IMAGE_CONFIG["count"],
                    },
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": BASE_IMAGE_CONFIG["width"],
                        "height": BASE_IMAGE_CONFIG["height"],
                        "prompts": [
                            {"text": prompt_text, "weight": 1.0},
                            {"text": BASE_IMAGE_NEGATIVE_PROMPT, "weight": -1.0},
                        ],
                        "sampler": BASE_IMAGE_CONFIG["sampler"],
                        "sdVae": "Automatic",
                        "steps": BASE_IMAGE_CONFIG["steps"],
                        "sd_model": BASE_IMAGE_CONFIG["model_id"],
                        "clip_skip": 2,
                        "cfg_scale": BASE_IMAGE_CONFIG["cfg_scale"],
                    },
                },
            ],
        }

        # Submit job
        job_id = submit_tensorart_job({}, payload)

        # Poll job status
        return poll_tensorart_job({}, job_id, output_directory)

    except Exception as e:
        raise TensorArtError(f"Failed to generate base image: {str(e)}")


# --- RunwayML API Functions ---


@with_retry(config=RUNWAYML_RETRY_CONFIG)
def create_runway_task(
    client: RunwayClient, image_base64: str, animation_prompt_text: str, seed: int
) -> str:
    """
    Create a RunwayML task.

    Args:
        client: RunwayML client
        image_base64: Base64 encoded image
        animation_prompt_text: Animation prompt
        seed: Random seed

    Returns:
        str: Task ID

    Raises:
        RunwayMLError: If task creation fails
    """
    try:
        # Get config values with type safety
        model = ANIMATION_CONFIG["model"]
        duration = ANIMATION_CONFIG["duration"]
        ratio = ANIMATION_CONFIG["ratio"]

        task_id = client.create_animation_task(
            image_base64=image_base64,
            prompt_text=animation_prompt_text,
            model=model,
            duration=duration,
            ratio=ratio,
            seed=seed,
        )
        return task_id
    except Exception as e:
        raise RunwayMLError(f"Failed to create RunwayML task: {str(e)}")


@with_retry(config=RetryConfig(max_retries=15, base_delay=10.0))
def poll_runway_task(
    client: RunwayClient,
    task_id: str,
    output_directory: Union[str, Path],
    output_filename_base: str,
    seed: int,
) -> Optional[str]:
    """
    Poll RunwayML task status until completion.

    Args:
        client: RunwayML client
        task_id: Task ID
        output_directory: Directory to save output
        output_filename_base: Base name for output file
        seed: Random seed

    Returns:
        Optional[str]: Path to saved video

    Raises:
        RunwayMLError: If polling fails
    """
    try:
        task_status = client.wait_for_task_completion(task_id)

        if task_status["status"] == "SUCCEEDED":
            video_url = client.extract_video_url(task_status)
            if not video_url:
                raise RunwayMLError("No video URL found in task status")

            # Download video
            output_path = (
                Path(output_directory) / f"{output_filename_base}_seed{seed}.mp4"
            )
            download_file(video_url, output_path)

            return str(output_path)
        else:
            raise RunwayMLError(f"Task failed with status: {task_status['status']}")

    except Exception as e:
        raise RunwayMLError(f"Failed to poll RunwayML task: {str(e)}")


def generate_animation_runway(
    base_image_path: Union[str, Path],
    animation_prompt_text: str,
    output_directory: Union[str, Path],
    output_filename_base: str,
    seed: int,
) -> Optional[str]:
    """
    Generate an animation using RunwayML.

    Args:
        base_image_path: Path to base image
        animation_prompt_text: Animation prompt
        output_directory: Directory to save output
        output_filename_base: Base name for output file
        seed: Random seed

    Returns:
        Optional[str]: Path to saved video

    Raises:
        RunwayMLError: If generation fails
    """
    logger.info("\n--- Starting Animation Generation (RunwayML) ---")

    if not runway_client.api_key:
        raise ConfigurationError("RunwayML API secret missing")

    try:
        # Initialize RunwayML client
        client = runway_client

        # Encode base image
        image_base64 = encode_image_base64(base_image_path)

        # Create task
        task_id = create_runway_task(client, image_base64, animation_prompt_text, seed)

        # Poll task status
        return poll_runway_task(
            client, task_id, output_directory, output_filename_base, seed
        )

    except Exception as e:
        raise RunwayMLError(f"Failed to generate animation: {str(e)}")


def generate_all_base_images(
    output_directory: Union[str, Path],
) -> List[Dict[str, str]]:
    """
    Generate base images for all countries.

    Args:
        output_directory: Directory to save output

    Returns:
        List[Dict[str, str]]: List of generated image info

    Raises:
        WorkflowError: If generation fails
    """
    logger.info("\n--- Generating Base Images for All Countries ---")

    generated_images = []
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Load country data
    country_manager.load_countries()
    countries = country_manager.get_all_countries()

    for country in countries:
        country_name = country.name.lower()  # Use lowercase name as ID
        logger.info(f"\nGenerating base image for {country.name}...")
        output_path, error = generate_base_image_tensorart(
            output_directory, country_name
        )

        if error:
            logger.error(f"Failed to generate base image for {country.name}: {error}")
            continue

        generated_images.append(
            {
                "country_code": country_name,
                "prompt_id": country_name,
                "image_path": str(output_path),
            }
        )

    return generated_images


def generate_all_animations(
    base_image_path: Union[str, Path],
    output_directory: Union[str, Path],
    start_seed: Optional[int] = None,
) -> List[str]:
    """
    Generate animations for all prompts.

    Args:
        base_image_path: Path to base image
        output_directory: Directory to save output
        start_seed: Optional starting seed

    Returns:
        List[str]: List of generated animation paths

    Raises:
        WorkflowError: If generation fails
    """
    logger.info("\n--- Generating Animations for All Prompts ---")

    generated_animations = []
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Initialize seed with type safety
    seed = start_seed if start_seed is not None else ANIMATION_CONFIG["seed_start"]

    for prompt in ANIMATION_PROMPTS:
        prompt_id = prompt["id"]
        prompt_text = prompt["text"]

        logger.info(f"\nGenerating animation for prompt {prompt_id}...")
        animation_path = generate_animation_runway(
            base_image_path, prompt_text, output_directory, prompt_id, seed
        )

        if animation_path is None:
            logger.error(f"Failed to generate animation for {prompt_id}")
            continue

        generated_animations.append(str(animation_path))
        seed += 1

    return generated_animations


def main(output_directory: Optional[Union[str, Path]] = None) -> None:
    """
    Main function for testing generation.

    Args:
        output_directory: Optional output directory
    """
    if not output_directory:
        output_directory = Path.cwd() / "output"

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Check credentials
    if not check_credentials():
        logger.error("API credentials check failed")
        return

    # Generate base images
    generated_images = generate_all_base_images(output_directory)
    if not generated_images:
        logger.error("No base images were generated")
        return

    # Get seed start with type safety
    start_seed = ANIMATION_CONFIG["seed_start"]

    # Generate animations for each base image
    for image_info in generated_images:
        image_path = image_info["image_path"]
        country_code = image_info["country_code"]

        logger.info(f"\nGenerating animations for {country_code}...")
        generated_animations = generate_all_animations(
            image_path, output_directory / country_code,
            start_seed=start_seed
        )

        if not generated_animations:
            logger.error(f"No animations were generated for {country_code}")
            continue

        logger.info(
            f"Successfully generated {len(generated_animations)} animations for {country_code}"
        )


if __name__ == "__main__":
    main()
