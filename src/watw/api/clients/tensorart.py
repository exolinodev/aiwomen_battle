"""
TensorArt API client for Women Around The World.

This module provides a client for interacting with the TensorArt API
for image generation.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, cast

import requests

from watw.api.clients.base import APIError, APITimeoutError, BaseAPIClient
from watw.api.config import Config


class TensorArtClient(BaseAPIClient):
    """
    Client for interacting with the TensorArt API.

    This client provides methods for generating images using
    the TensorArt API.
    """

    def __init__(self, require_api_key: bool = True):
        """
        Initialize the TensorArt API client.

        Args:
            require_api_key: Whether to require a valid API key
        """
        config = Config()
        api_key = config.get_setting("tensorart", "api_key", "")
        base_url = config.get_setting(
            "tensorart", "base_url", "https://tams.tensor.art"
        )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            service_name="tensorart",
            require_api_key=require_api_key,
        )

        self.config = config
        # Load default settings
        self.model_id = self.config.get_setting("tensorart", "model_id")
        self.base_image_width = self.config.get_setting(
            "tensorart", "base_image_width", 768
        )
        self.base_image_height = self.config.get_setting(
            "tensorart", "base_image_height", 1280
        )
        self.base_image_steps = self.config.get_setting(
            "tensorart", "base_image_steps", 30
        )
        self.base_image_cfg_scale = self.config.get_setting(
            "tensorart", "base_image_cfg_scale", 7
        )
        self.base_image_sampler = self.config.get_setting(
            "tensorart", "base_image_sampler", "Euler"
        )

        # Set endpoints
        self.submit_job_endpoint = "/api/generate"

        # Mask API key for logging
        masked_key = (
            self.api_key[:8] + "*" * (len(self.api_key) - 8) if self.api_key else None
        )
        self.logger.info(f"Initialized TensorArt client with API key: {masked_key}")

    def make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the TensorArt API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint
            data: Request data
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data as dictionary

        Raises:
            APIError: If request fails
        """
        if not headers:
            headers = {}

        # Add Bearer token authentication
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Add content type header
        headers["Content-Type"] = "application/json"

        try:
            response = requests.request(
                method=method,
                url=f"{self.base_url}{endpoint}",
                json=data,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(f"Request failed: {str(e)}")

    def download_file(self, url: str, output_path: Union[str, Path]) -> Path:
        """
        Download a file from a URL.

        Args:
            url: URL to download from
            output_path: Path to save the file

        Returns:
            Path to the downloaded file

        Raises:
            APIError: If download fails
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path
        except Exception as e:
            self.logger.error(f"Failed to download file: {str(e)}")
            raise APIError(f"Failed to download file: {str(e)}")

    def get_job_status_endpoint(self, job_id: str) -> str:
        """
        Get the endpoint for checking job status.

        Args:
            job_id: ID of the job

        Returns:
            Endpoint URL
        """
        return f"{self.base_url}/v1/jobs/{job_id}"

    def create_job(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        sampler: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Create a new image generation job.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Negative prompt for image generation
            width: Image width
            height: Image height
            steps: Number of inference steps
            cfg_scale: Guidance scale
            sampler: Sampler to use
            seed: Random seed

        Returns:
            Job ID
        """
        payload = {
            "text": prompt,
            "negative_text": negative_prompt,
            "width": width or self.base_image_width,
            "height": height or self.base_image_height,
            "steps": steps or self.base_image_steps,
            "guidance_scale": cfg_scale or self.base_image_cfg_scale,
            "sampler": sampler or self.base_image_sampler,
            "seed": seed if seed is not None else -1,
            "num_outputs": 1,
            "safety_check": True,
            "enhance_prompt": False,
            "model": self.model_id
        }

        try:
            response = self.post("/artworks/txt2img", json=payload)
            return response["artwork_id"]
        except Exception as e:
            raise APIError(f"Failed to create job: {str(e)}")

    def check_job_status(self, job_id: str) -> Tuple[str, Optional[str]]:
        """
        Check the status of a job.

        Args:
            job_id: The ID of the job to check

        Returns:
            A tuple of (status, image_url) where status is one of:
            'pending', 'processing', 'completed', 'failed'
            and image_url is the URL of the generated image if completed
        """
        try:
            response = self.get(f"/artworks/{job_id}")
            status = response.get("status", "").lower()
            
            if status == "completed":
                image_url = response.get("url")
                return "completed", image_url
            elif status == "processing":
                return "processing", None
            elif status == "failed":
                return "failed", None
            else:
                return "pending", None

        except Exception as e:
            raise APIError(f"Failed to check job status: {str(e)}")

    def wait_for_job_completion(
        self, job_id: str, timeout: int = 300, polling_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Wait for a job to complete.

        Args:
            job_id: ID of the job
            timeout: Maximum time to wait in seconds
            polling_interval: Time between polls in seconds

        Returns:
            Final job status

        Raises:
            APITimeoutError: If the job times out
            APIError: If the job fails
        """
        start_time = time.time()
        attempts = 0
        max_attempts = timeout // polling_interval

        self.logger.info(f"Waiting for TensorArt job {job_id} to complete")

        while attempts < max_attempts:
            attempts += 1
            self.logger.debug(f"Polling attempt {attempts}/{max_attempts}")

            try:
                # Wait before polling
                if attempts > 1:
                    time.sleep(polling_interval)

                # Check job status
                job_status = self.check_job_status(job_id)
                status = job_status[0]

                # Check for completion
                if status == "completed":
                    self.logger.info(f"TensorArt job {job_id} completed successfully")
                    return {"status": status, "image_url": job_status[1]}
                elif status == "failed":
                    # Extract error message
                    error_details = (
                        job_status[1]
                        or "Unknown error"
                    )
                    self.logger.error(f"TensorArt job {job_id} failed: {error_details}")
                    raise APIError(f"TensorArt job failed: {error_details}")
                else:
                    # Still processing
                    self.logger.debug(
                        f"TensorArt job {job_id} still in progress. Status: {status}"
                    )
                    continue

            except APIError as e:
                # Only propagate errors if they are fatal
                if "failed" in str(e) or "error" in str(e):
                    raise
                self.logger.warning(
                    f"Error polling job status: {str(e)}. Continuing to poll..."
                )
                continue

        # If we get here, the job timed out
        elapsed = time.time() - start_time
        self.logger.error(
            f"TensorArt job {job_id} timed out after {elapsed:.1f} seconds"
        )
        raise APITimeoutError(service=self.service_name, timeout=timeout)

    def extract_image_url(self, job_status: Dict[str, Any]) -> str:
        """
        Extract the image URL from a job status response.

        Args:
            job_status: Job status response from TensorArt API

        Returns:
            URL of the generated image

        Raises:
            APIError: If image URL cannot be extracted
        """
        if not isinstance(job_status, dict):
            raise APIError("Invalid job status format")

        # Check for images in the response
        images = job_status.get("images")
        if not isinstance(images, list) or not images:
            self.logger.error("No images found in job status")
            raise APIError("No images found in job status")

        # Get the first image
        first_image = images[0]
        if not isinstance(first_image, dict):
            self.logger.error("Invalid image format in job status")
            raise APIError("Invalid image format in job status")

        # Extract the URL
        image_url = first_image.get("url")
        if not isinstance(image_url, str) or not image_url:
            self.logger.error("No valid image URL found in job status")
            raise APIError("No valid image URL found in job status")

        self.logger.info(f"Found image URL: {image_url}")
        return image_url

    def generate_image(
        self,
        prompt_text: str,
        output_path: Union[str, Path],
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        sampler: Optional[str] = None,
        model_id: Optional[str] = None,
        seed: int = -1,
        timeout: int = 300,
        polling_interval: int = 10,
    ) -> Tuple[Path, str]:
        """
        Generate an image and save it to a file.

        Args:
            prompt_text: Text prompt for the image
            output_path: Path to save the image
            negative_prompt: Negative prompt to avoid certain elements
            width: Width of the image
            height: Height of the image
            steps: Number of diffusion steps
            cfg_scale: Guidance scale
            sampler: Sampler to use
            model_id: ID of the model to use
            seed: Random seed for generation (-1 for random)
            timeout: Maximum time to wait in seconds
            polling_interval: Time between polls in seconds

        Returns:
            Tuple of (path to saved image, image URL)

        Raises:
            APITimeoutError: If the generation times out
            APIError: If the generation fails
        """
        try:
            # Convert output path to Path object
            output_path = Path(output_path)

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create the job
            job_id = self.create_job(
                prompt=prompt_text,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                seed=seed,
            )

            # Wait for job completion
            job_status = self.wait_for_job_completion(
                job_id=job_id, timeout=timeout, polling_interval=polling_interval
            )

            # Extract image URL
            image_url = self.extract_image_url(job_status)

            # Download the image
            downloaded_path = self.download_file(image_url, output_path)

            self.logger.info(f"Image downloaded to {downloaded_path}")
            return downloaded_path, image_url

        except Exception as e:
            if not isinstance(e, (APIError, APITimeoutError)):
                self.logger.error(f"Error generating image: {str(e)}")
                raise APIError(f"Failed to generate image: {str(e)}")
            raise
