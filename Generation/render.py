"""
Image and animation generation module.

This module provides functions for generating base images using TensorArt API
and creating animations using RunwayML API. It includes utilities for API
interaction, file operations, and workflow management.
"""

import os
import requests
import json
import time
import uuid
import base64
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from runwayml import RunwayML
from enum import Enum
import sys
import random

from Generation.countries import COUNTRIES, BASE_IMAGE_NEGATIVE_PROMPT, ANIMATION_NEGATIVE_PROMPT
from watw.utils.common.exceptions import (
    ConfigurationError,
    APIError,
    TensorArtError,
    RunwayMLError,
    FileOperationError,
    WorkflowError,
    ValidationError
)
from watw.utils.common.credentials import credentials
from watw.utils.common.rate_limiter import (
    with_retry,
    RetryConfig,
    RateLimitExceeded,
    handle_rate_limit,
    extract_retry_after
)
from watw.utils.common.api_validator import (
    validate_tensorart_job_response,
    validate_tensorart_job_status_response,
    validate_runway_task_response,
    validate_runway_task_status_response,
    ValidationError
)

# Configure logging
logger = logging.getLogger(__name__)

# --- Constants and Configuration ---

class TaskStatus(Enum):
    """Enumeration of possible task statuses."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

# API Endpoints
TENSORART_API_BASE_URL = "https://api.tensorart.ai"
RUNWAYML_API_BASE_URL = "https://api.runwayml.com"

# Base Image Generation Parameters (TensorArt)
BASE_IMAGE_CONFIG = {
    'width': 768,
    'height': 1280,
    'steps': 30,
    'cfg_scale': 7,
    'sampler': "Euler",
    'model_id': "757279507095956705",
    'seed': -1,
    'count': 1
}

# Animation Parameters (RunwayML)
ANIMATION_CONFIG = {
    'model': "gen3a_turbo",
    'duration': 5,
    'ratio': "768:1280",
    'seed_start': random.randint(1, 1000000)
}

# Retry Configurations
TENSORART_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=30.0,
    max_delay=300.0,
    jitter=True,
    exponential_backoff=True
)

RUNWAYML_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    base_delay=60.0,
    max_delay=300.0,
    jitter=True,
    exponential_backoff=True
)

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
        credentials.get_tensorart_token()
        credentials.get_runwayml_secret()
        logger.info("API credentials loaded successfully")
        return True
    except ConfigurationError as e:
        logger.error(f"Credential error: {str(e)}")
        raise

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
        
        response = requests.get(url, stream=True, timeout=180)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"Successfully downloaded {save_path}")
        return save_path
        
    except requests.exceptions.RequestException as e:
        raise FileOperationError(f"Failed to download file from {url}: {str(e)}")
    except Exception as e:
        raise FileOperationError(f"Unexpected error downloading file: {str(e)}")

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
        from PIL import Image
        import io
        
        with Image.open(image_path) as img:
            # Validate and resize if needed
            width, height = img.size
            if width != BASE_IMAGE_CONFIG['width'] or height != BASE_IMAGE_CONFIG['height']:
                logger.warning(f"Image dimensions {width}x{height} do not match required {BASE_IMAGE_CONFIG['width']}x{BASE_IMAGE_CONFIG['height']}")
                img = img.resize((BASE_IMAGE_CONFIG['width'], BASE_IMAGE_CONFIG['height']), Image.Resampling.LANCZOS)
                logger.info("Image has been resized")
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            return base64.b64encode(img_byte_arr).decode('utf-8')
            
    except Exception as e:
        raise FileOperationError(f"Failed to encode image {image_path}: {str(e)}")

# --- TensorArt API Functions ---

def get_tensorart_job_status_endpoint(job_id: str) -> str:
    """Get the endpoint URL for checking TensorArt job status."""
    return f"{TENSORART_API_BASE_URL}/v1/jobs/{job_id}"

@with_retry(config=TENSORART_RETRY_CONFIG)
def submit_tensorart_job(headers: Dict[str, str], payload: Dict[str, Any]) -> str:
    """
    Submit a job to TensorArt API.
    
    Args:
        headers: Request headers
        payload: Job parameters
        
    Returns:
        str: Job ID
        
    Raises:
        TensorArtError: If submission fails
        RateLimitExceeded: If rate limit is exceeded
        ValidationError: If response validation fails
    """
    try:
        response = requests.post(
            f"{TENSORART_API_BASE_URL}/v1/jobs",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 429:
            handle_rate_limit(response)
            
        validation_result = validate_tensorart_job_response(response)
        if not validation_result.is_valid:
            error_messages = [str(e) for e in validation_result.errors]
            raise ValidationError(f"Invalid TensorArt job response: {'; '.join(error_messages)}")
            
        return validation_result.value['job']['id']
        
    except Exception as e:
        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
            retry_after = extract_retry_after(e.response)
            if retry_after:
                raise RateLimitExceeded("Rate limit exceeded", retry_after)
            else:
                raise RateLimitExceeded("Rate limit exceeded")
                
        raise TensorArtError(f"Job submission failed: {str(e)}",
                           status_code=getattr(e.response, 'status_code', None),
                           response_data=getattr(e.response, 'json', lambda: None)())

@with_retry(config=RetryConfig(max_retries=10, base_delay=10.0))
def poll_tensorart_job(headers: Dict[str, str], job_id: str, output_directory: Union[str, Path]) -> Tuple[Optional[Path], Optional[str]]:
    """
    Poll TensorArt job until completion.
    
    Args:
        headers: Request headers
        job_id: Job ID to poll
        output_directory: Output directory
        
    Returns:
        Tuple[Optional[Path], Optional[str]]: Path to saved image and image URL
        
    Raises:
        TensorArtError: If polling fails
        RateLimitExceeded: If rate limit is exceeded
        ValidationError: If response validation fails
    """
    polling_endpoint = get_tensorart_job_status_endpoint(job_id)
    polling_interval = 10
    max_attempts = 30
    
    for attempt in range(max_attempts):
        try:
            time.sleep(polling_interval)
            response = requests.get(polling_endpoint, headers=headers, timeout=30)
            
            if response.status_code == 429:
                handle_rate_limit(response)
                
            validation_result = validate_tensorart_job_status_response(response)
            if not validation_result.is_valid:
                error_messages = [str(e) for e in validation_result.errors]
                raise ValidationError(f"Invalid TensorArt job status response: {'; '.join(error_messages)}")
                
            job_status_data = validation_result.value
            status = job_status_data.get('job', {}).get('status', '').upper()
            
            if status == TaskStatus.SUCCEEDED.value:
                image_url = job_status_data.get('job', {}).get('output', {}).get('images', [{}])[0].get('url')
                if not image_url:
                    raise TensorArtError("No image URL in successful job response")
                    
                output_path = Path(output_directory) / f"base_image_{job_id}.png"
                downloaded_path = download_file(image_url, output_path)
                return downloaded_path, image_url
                
            elif status == TaskStatus.FAILED.value:
                error_message = job_status_data.get('job', {}).get('error', 'Unknown error')
                raise TensorArtError(f"Job failed: {error_message}")
                
            logger.info(f"Job status: {status} (attempt {attempt + 1}/{max_attempts})")
            
        except Exception as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                retry_after = extract_retry_after(e.response)
                if retry_after:
                    raise RateLimitExceeded("Rate limit exceeded", retry_after)
                else:
                    raise RateLimitExceeded("Rate limit exceeded")
                    
            logger.warning(f"Polling attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_attempts - 1:
                raise TensorArtError(f"Maximum polling retries reached: {str(e)}")
            continue
            
    raise TensorArtError("Job polling timed out")

def generate_base_image_tensorart(
    output_directory: Union[str, Path],
    prompt_id: Optional[str] = None,
    prompt_text: Optional[str] = None
) -> Tuple[Optional[Path], Optional[str]]:
    """
    Generate a base image using TensorArt API.
    
    Args:
        output_directory: Output directory
        prompt_id: Optional prompt ID
        prompt_text: Optional prompt text
        
    Returns:
        Tuple[Optional[Path], Optional[str]]: Path to saved image and image URL
        
    Raises:
        TensorArtError: If generation fails
        ConfigurationError: If credentials are missing
        FileOperationError: If file operations fail
    """
    logger.info("Starting Base Image Generation (TensorArt)")
    
    token = credentials.get_tensorart_token()
    
    if prompt_id is None or prompt_text is None:
        prompt = COUNTRIES[0]["base_image_prompt"]
        prompt_id = COUNTRIES[0]["id"]
        prompt_text = prompt
    
    logger.info(f"Generating base image for: {prompt_id}")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=UTF-8",
        "Accept": "application/json"
    }
    
    payload = {
        "request_id": str(uuid.uuid4()),
        "stages": [
            {
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": BASE_IMAGE_CONFIG['seed'],
                    "count": BASE_IMAGE_CONFIG['count']
                }
            },
            {
                "type": "DIFFUSION",
                "diffusion": {
                    "width": BASE_IMAGE_CONFIG['width'],
                    "height": BASE_IMAGE_CONFIG['height'],
                    "prompts": [
                        {"text": prompt_text, "weight": 1.0},
                        {"text": BASE_IMAGE_NEGATIVE_PROMPT, "weight": -1.0}
                    ],
                    "sampler": BASE_IMAGE_CONFIG['sampler'],
                    "sdVae": "Automatic",
                    "steps": BASE_IMAGE_CONFIG['steps'],
                    "sd_model": BASE_IMAGE_CONFIG['model_id'],
                    "clip_skip": 2,
                    "cfg_scale": BASE_IMAGE_CONFIG['cfg_scale']
                }
            }
        ]
    }
    
    try:
        job_id = submit_tensorart_job(headers, payload)
        logger.info(f"TensorArt job submitted successfully. Job ID: {job_id}")
        return poll_tensorart_job(headers, job_id, output_directory)
        
    except TensorArtError as e:
        logger.error(f"TensorArt operation failed: {str(e)}")
        if e.status_code:
            logger.error(f"Status code: {e.status_code}")
        if e.response_data:
            logger.error(f"Response data: {json.dumps(e.response_data, indent=2)}")
        return None, None

# --- RunwayML API Functions ---

@with_retry(config=RUNWAYML_RETRY_CONFIG)
def create_runway_task(
    client: RunwayML,
    image_base64: str,
    animation_prompt_text: str,
    seed: int
) -> str:
    """
    Create a RunwayML animation task.
    
    Args:
        client: RunwayML client
        image_base64: Base64 encoded image
        animation_prompt_text: Animation prompt
        seed: Random seed
        
    Returns:
        str: Task ID
        
    Raises:
        RunwayMLError: If task creation fails
        RateLimitExceeded: If rate limit is exceeded
        ValidationError: If response validation fails
    """
    try:
        enhanced_prompt = f"Create a smooth, cinematic animation. {animation_prompt_text} Maintain consistent motion and high quality throughout the {ANIMATION_CONFIG['duration']}-second duration."
        
        task_response = client.image_to_video.create(
            model=ANIMATION_CONFIG['model'],
            prompt_image=f"data:image/png;base64,{image_base64}",
            prompt_text=enhanced_prompt,
            duration=ANIMATION_CONFIG['duration'],
            ratio=ANIMATION_CONFIG['ratio'],
            seed=seed,
            watermark=False
        )
        
        validation_result = validate_runway_task_response(task_response)
        if not validation_result.is_valid:
            error_messages = [str(e) for e in validation_result.errors]
            raise ValidationError(f"Invalid RunwayML task response: {'; '.join(error_messages)}")
            
        return validation_result.value['id']
        
    except Exception as e:
        if hasattr(e, 'response') and e.response and e.response.status_code == 429:
            retry_after = extract_retry_after(e.response)
            if retry_after:
                raise RateLimitExceeded("Rate limit exceeded", retry_after)
            else:
                raise RateLimitExceeded("Rate limit exceeded")
                
        raise RunwayMLError(f"Task creation failed: {str(e)}",
                          status_code=getattr(e.response, 'status_code', None),
                          response_data=getattr(e.response, 'json', lambda: None)())

@with_retry(config=RetryConfig(max_retries=15, base_delay=10.0))
def poll_runway_task(
    client: RunwayML,
    task_id: str,
    output_directory: Union[str, Path],
    output_filename_base: str,
    seed: int
) -> Optional[str]:
    """
    Poll RunwayML task until completion.
    
    Args:
        client: RunwayML client
        task_id: Task ID
        output_directory: Output directory
        output_filename_base: Base filename
        seed: Random seed
        
    Returns:
        Optional[str]: Path to generated video
        
    Raises:
        RunwayMLError: If polling fails
        RateLimitExceeded: If rate limit is exceeded
        ValidationError: If response validation fails
    """
    max_poll_retries = 30
    poll_retry_delay = 10
    
    for attempt in range(max_poll_retries):
        try:
            task = client.image_to_video.get(task_id)
            
            validation_result = validate_runway_task_status_response(task)
            if not validation_result.is_valid:
                error_messages = [str(e) for e in validation_result.errors]
                raise ValidationError(f"Invalid RunwayML task status response: {'; '.join(error_messages)}")
            
            status = task.status.upper()
            
            if status == TaskStatus.SUCCEEDED.value:
                output_path = Path(output_directory) / f"{output_filename_base}_{seed}.mp4"
                video_url = task.output.get('video_url')
                
                if not video_url:
                    raise RunwayMLError("No video URL in successful task response")
                    
                downloaded_path = download_file(video_url, output_path)
                logger.info(f"Animation saved to: {downloaded_path}")
                return str(downloaded_path)
                
            elif status == TaskStatus.FAILED.value:
                error_message = getattr(task, 'error', 'Unknown error')
                raise RunwayMLError(f"Task failed: {error_message}")
                
            logger.info(f"Task status: {status} (attempt {attempt + 1}/{max_poll_retries})")
            time.sleep(poll_retry_delay)
            
        except Exception as e:
            if hasattr(e, 'response') and e.response and e.response.status_code == 429:
                retry_after = extract_retry_after(e.response)
                if retry_after:
                    raise RateLimitExceeded("Rate limit exceeded", retry_after)
                else:
                    raise RateLimitExceeded("Rate limit exceeded")
                    
            logger.warning(f"Polling attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_poll_retries - 1:
                raise RunwayMLError(f"Maximum polling retries reached: {str(e)}")
            continue
            
    raise RunwayMLError("Task polling timed out")

def generate_animation_runway(
    base_image_path: Union[str, Path],
    animation_prompt_text: str,
    output_directory: Union[str, Path],
    output_filename_base: str,
    seed: int
) -> Optional[str]:
    """
    Generate an animation using RunwayML API.
    
    Args:
        base_image_path: Path to base image
        animation_prompt_text: Animation prompt
        output_directory: Output directory
        output_filename_base: Base filename
        seed: Random seed
        
    Returns:
        Optional[str]: Path to generated video
        
    Raises:
        RunwayMLError: If generation fails
        ConfigurationError: If credentials are missing
        FileOperationError: If file operations fail
    """
    logger.info(f"Starting Animation Generation for: {output_filename_base}")
    
    api_secret = credentials.get_runwayml_secret()
    
    if not base_image_path:
        raise ValidationError("Base image path is required")
    if not animation_prompt_text:
        raise ValidationError("Animation prompt text is required")
        
    try:
        client = RunwayML(api_key=api_secret)
        image_base64 = encode_image_base64(base_image_path)
        
        task_id = create_runway_task(client, image_base64, animation_prompt_text, seed)
        logger.info(f"RunwayML task submitted successfully. Task ID: {task_id}")
        
        return poll_runway_task(client, task_id, output_directory, output_filename_base, seed)
        
    except (RunwayMLError, FileOperationError) as e:
        logger.error(f"RunwayML operation failed: {str(e)}")
        if hasattr(e, 'status_code') and e.status_code:
            logger.error(f"Status code: {e.status_code}")
        if hasattr(e, 'response_data') and e.response_data:
            logger.error(f"Response data: {json.dumps(e.response_data, indent=2)}")
        return None

# --- Workflow Functions ---

def generate_all_base_images(output_directory: Union[str, Path]) -> List[Dict[str, str]]:
    """
    Generate base images for all countries.
    
    Args:
        output_directory: Output directory
        
    Returns:
        List[Dict[str, str]]: List of generated image info
        
    Raises:
        WorkflowError: If generation fails
    """
    logger.info("Generating All Base Images")
    base_images = []
    
    for country_key, country_data in COUNTRIES.items():
        prompt_id = country_data["id"]
        prompt_text = country_data["base_image_prompt"]
        
        logger.info(f"Generating base image for: {prompt_id}")
        base_image_path, _ = generate_base_image_tensorart(
            output_directory=output_directory,
            prompt_id=prompt_id,
            prompt_text=prompt_text
        )
        
        if base_image_path:
            base_images.append({
                "id": prompt_id,
                "path": str(base_image_path)
            })
            logger.info(f"Base image for {prompt_id} saved to: {base_image_path}")
        else:
            logger.error(f"Failed to generate base image for: {prompt_id}")
    
    return base_images

def generate_all_animations(
    base_image_path: Union[str, Path],
    output_directory: Union[str, Path],
    start_seed: Optional[int] = None
) -> List[str]:
    """
    Generate animations for all countries.
    
    Args:
        base_image_path: Path to base image
        output_directory: Output directory
        start_seed: Optional starting seed
        
    Returns:
        List[str]: List of generated video paths
        
    Raises:
        WorkflowError: If generation fails
    """
    logger.info("Generating All Animations")
    generated_clips = []
    current_seed = start_seed or ANIMATION_CONFIG['seed_start']
    
    for country_key, country_data in COUNTRIES.items():
        try:
            prompt_id = country_data["id"]
            prompt_text = country_data["animation_prompt"]
            output_filename_base = f"animation_{prompt_id}"
            
            video_path = generate_animation_runway(
                base_image_path=base_image_path,
                animation_prompt_text=prompt_text,
                output_directory=output_directory,
                output_filename_base=output_filename_base,
                seed=current_seed
            )
            
            if video_path:
                generated_clips.append(video_path)
            else:
                logger.error(f"Failed to generate animation for prompt: {prompt_id}")
                
            current_seed += 1
            
        except Exception as e:
            logger.error(f"Error generating animation for {country_key}: {str(e)}")
            continue
    
    return generated_clips

def main():
    """
    Main workflow execution.
    
    Raises:
        WorkflowError: If workflow fails
        ConfigurationError: If configuration is missing
    """
    logger.info("Starting Gym Short Generation Workflow")
    
    try:
        check_credentials()
        
        # Generate base image
        base_image_local_path, base_image_url = generate_base_image_tensorart(OUTPUT_FOLDER)
        if not base_image_local_path or not base_image_url:
            raise WorkflowError("Base image generation failed")
            
        logger.info(f"Base image saved to: {base_image_local_path}")
        
        # User confirmation
        response = input("Do you want to continue with animation generation? (yes/no): ").lower().strip()
        if response != 'yes':
            logger.info("Animation generation cancelled by user")
            return
            
        # Generate animations
        generated_clips = generate_all_animations(
            base_image_path=base_image_local_path,
            output_directory=OUTPUT_FOLDER
        )
        
        # Workflow conclusion
        logger.info("--- Workflow Finished ---")
        if base_image_local_path:
            logger.info(f"Base image saved at: {base_image_local_path}")
        if generated_clips:
            logger.info("Generated animation clips:")
            for clip in generated_clips:
                logger.info(f"- {clip}")
            logger.info(f"All files are located in the '{OUTPUT_FOLDER}' directory")
        else:
            logger.warning("No animation clips were successfully generated")
            
    except Exception as e:
        raise WorkflowError(f"Workflow failed: {str(e)}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)