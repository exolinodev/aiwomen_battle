"""
RunwayML API client for Women Around The World.

This module provides a client for interacting with the RunwayML API
for video generation and animation.
"""

import base64
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast, Literal, List
from datetime import datetime, timedelta

try:
    from runwayml import RunwayML, NotGiven
except ImportError:
    # Fallback if NotGiven isn't directly available
    from runwayml import RunwayML
    from typing import Any as NotGiven  # type: ignore

from PIL import Image, ImageFilter

from watw.api.clients.base import APIError, APITimeoutError, BaseAPIClient
from watw.api.config import Config


class RunwayClient(BaseAPIClient):
    """
    Client for interacting with the RunwayML API.

    This client provides methods for generating videos using
    the RunwayML API.
    """

    def __init__(self, require_api_key: bool = True):
        """
        Initialize the RunwayML API client.

        Args:
            require_api_key: Whether to require a valid API key
        """
        config = Config()
        api_key = config.get_setting("runwayml", "api_key", "")
        base_url = config.get_setting(
            "runwayml", "base_url", "https://api.runwayml.com"
        )

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            service_name="runwayml",
            require_api_key=require_api_key,
        )

        self.config = config
        # Initialize the RunwayML client if API key is available
        self.client: Optional[RunwayML] = None
        if self.api_key:
            try:
                self.client = RunwayML(api_key=self.api_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize RunwayML client: {e}")
                # Continue without the client - methods will check self.client before use

        # Load default settings
        self.animation_model = self.config.get_setting(
            "runwayml", "animation_model", "gen3a_turbo"
        )
        self.animation_duration = self.config.get_setting(
            "runwayml", "animation_duration", 5
        )
        self.animation_ratio = self.config.get_setting(
            "runwayml", "animation_ratio", "768:1280"
        )

        # Mask API key for logging
        masked_key = (
            self.api_key[:8] + "*" * (len(self.api_key) - 8) if self.api_key else None
        )
        self.logger.info(f"Initialized RunwayML client with API key: {masked_key}")

    def encode_image_base64(self, image_path: Union[str, Path]) -> str:
        """
        Encode an image file as a base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded image string

        Raises:
            APIError: If encoding fails
        """
        try:
            import io

            # Convert path to Path object
            image_path = Path(image_path)

            # Open and validate the image
            with Image.open(image_path) as img:
                # Check dimensions
                width, height = img.size
                if width != 768 or height != 1280:
                    self.logger.warning(
                        f"Image dimensions {width}x{height} do not match required 768x1280"
                    )
                    # Resize image to match requirements
                    try:
                        resample_filter = Image.Resampling.LANCZOS
                    except AttributeError:
                        # Fallback for older Pillow versions
                        resample_filter = Image.LANCZOS
                    img = img.resize((768, 1280), resample_filter)
                    self.logger.info("Image has been resized to 768x1280")

                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Save to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr_bytes = img_byte_arr.getvalue()

                return base64.b64encode(img_byte_arr_bytes).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image: {str(e)}")
            raise APIError(f"Failed to encode image {image_path}: {str(e)}")

    def create_animation_task(
        self,
        image_base64: str,
        prompt_text: str,
        model: Literal["gen3a_turbo"],
        duration: Literal[5, 10],
        ratio: Literal["1280:768", "768:1280"],
        seed: Optional[int] = None,
        watermark: bool = False,
    ) -> str:
        """
        Create an animation task using the RunwayML API.

        Args:
            image_base64: Base64-encoded image
            prompt_text: Text prompt for the animation
            model: Model to use (must be "gen3a_turbo")
            duration: Duration in seconds (must be 5 or 10)
            ratio: Aspect ratio (must be "1280:768" or "768:1280")
            seed: Random seed
            watermark: Whether to add watermark

        Returns:
            Task ID

        Raises:
            APIError: If task creation fails
        """
        if not self.client:
            raise APIError("RunwayML client not initialized")

        try:
            # Create the task
            task = self.client.image_to_video.create(
                model=model,
                prompt_image=f"data:image/png;base64,{image_base64}",
                prompt_text=prompt_text,
                duration=duration,
                ratio=ratio,
                seed=seed if seed is not None else NotGiven(),
                watermark=watermark,
            )
            return task.id
        except Exception as e:
            self.logger.error(f"Error creating animation task: {str(e)}")
            raise APIError(f"Failed to create animation task: {str(e)}")

    def wait_for_task_completion(
        self, task_id: str, poll_interval: int = 10, timeout: int = 600
    ) -> Dict[str, Any]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task ID
            poll_interval: Interval between checks in seconds
            timeout: Maximum time to wait in seconds

        Returns:
            Task status

        Raises:
            APIError: If task fails
            APITimeoutError: If task times out
        """
        if not self.client:
            raise APIError("RunwayML client not initialized")

        start_time = time.time()
        while True:  # Loop will exit via return or raise
            if time.time() - start_time > timeout:
                raise APITimeoutError(service=self.service_name, timeout=timeout)

            try:
                task = self.client.tasks.retrieve(task_id)
                if task.status == "completed":
                    return task.to_dict()
                elif task.status == "failed":
                    raise APIError(f"Task failed: {task.error}")
                time.sleep(poll_interval)
            except Exception as e:
                self.logger.error(f"Error checking task status: {str(e)}")
                raise APIError(f"Failed to check task status: {str(e)}")

    def extract_video_url(self, task_status: Dict[str, Any]) -> str:
        """
        Extract video URL from task status.

        Args:
            task_status: Task status dictionary

        Returns:
            Video URL

        Raises:
            APIError: If URL extraction fails
        """
        try:
            video_url = task_status.get("output", {}).get("url")
            if isinstance(video_url, str) and video_url:
                return video_url
            raise APIError("No valid video URL found in task output")
        except Exception as e:
            self.logger.error(f"Error extracting video URL: {str(e)}")
            raise APIError(f"Failed to extract video URL: {str(e)}")

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
        if not self.client:
            raise APIError("RunwayML client not initialized")

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import requests
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            return output_path
        except Exception as e:
            self.logger.error(f"Error downloading file: {str(e)}")
            raise APIError(f"Failed to download file: {str(e)}")

    def generate_animation(
        self,
        image_path: Union[str, Path],
        prompt_text: str,
        output_path: Union[str, Path],
        seed: Optional[int] = None,
        model: Optional[str] = None,
        duration: Optional[int] = None,
        ratio: Optional[str] = None,
        watermark: bool = False,
        max_retries: int = 5,
        retry_delay: int = 60,
        poll_interval: int = 10,
        timeout: int = 600,
    ) -> Path:
        """
        Generate an animation using the RunwayML API.

        Args:
            image_path: Path to the input image
            prompt_text: Text prompt for the animation
            output_path: Path to save the output video
            seed: Random seed
            model: Model to use (defaults to config)
            duration: Duration in seconds (defaults to config)
            ratio: Aspect ratio (defaults to config)
            watermark: Whether to add watermark
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            poll_interval: Interval between status checks
            timeout: Maximum time to wait for completion

        Returns:
            Path to the generated video

        Raises:
            APIError: If generation fails
        """
        if not self.client:
            raise APIError("RunwayML client not initialized")

        # Use defaults from config if not specified
        model = model or self.animation_model
        duration = duration or self.animation_duration
        ratio = ratio or self.animation_ratio

        # Validate parameters
        allowed_models: List[Literal["gen3a_turbo"]] = ["gen3a_turbo"]
        allowed_durations: List[Literal[5, 10]] = [5, 10]
        allowed_ratios: List[Literal["1280:768", "768:1280"]] = ["1280:768", "768:1280"]

        validated_model = cast(Literal["gen3a_turbo"], model)
        validated_duration = cast(Literal[5, 10], duration)
        validated_ratio = cast(Literal["1280:768", "768:1280"], ratio)

        if validated_model not in allowed_models:
            raise ValueError(f"Invalid model '{model}'. Allowed: {allowed_models}")
        if validated_duration not in allowed_durations:
            raise ValueError(f"Invalid duration '{duration}'. Allowed: {allowed_durations}")
        if validated_ratio not in allowed_ratios:
            raise ValueError(f"Invalid ratio '{ratio}'. Allowed: {allowed_ratios}")

        # Convert paths to Path objects
        image_path = Path(image_path)
        output_path = Path(output_path)

        # Encode image
        image_base64 = self.encode_image_base64(image_path)

        # Retry loop for task creation
        for attempt in range(max_retries):
            try:
                # Enhance the prompt
                enhanced_prompt = f"Create a smooth, cinematic animation. {prompt_text} Maintain consistent motion and high quality throughout the {validated_duration}-second duration."

                # Create the task
                self.logger.info(
                    f"Creating animation task with prompt: {enhanced_prompt[:50]}..."
                )
                task_response = self.client.image_to_video.create(
                    model=validated_model,
                    prompt_image=f"data:image/png;base64,{image_base64}",
                    prompt_text=enhanced_prompt,
                    duration=validated_duration,
                    ratio=validated_ratio,
                    seed=seed if seed is not None else NotGiven(),
                    watermark=watermark,
                )
                break  # If successful, break the retry loop
            except Exception as e:
                if attempt == max_retries - 1:
                    raise APIError(f"Failed to create animation task after {max_retries} attempts: {str(e)}")
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        if task_response is None:  # type: ignore[unreachable]
            raise APIError("Failed to create animation task after retries")

        # Wait for task completion
        task_status = self.wait_for_task_completion(
            task_response.id, poll_interval=poll_interval, timeout=timeout
        )

        # Download the video
        video_url = self.extract_video_url(task_status)
        return self.download_file(video_url, output_path)

    def check_usage(self, days: int = 1) -> Dict[str, Any]:
        """
        Check API usage for the last N days.

        Args:
            days: Number of days to check

        Returns:
            Usage statistics

        Raises:
            APIError: If usage check fails
        """
        if not self.client:
            raise APIError("RunwayML client not initialized")

        try:
            # Initialize status counts
            status_counts: Dict[str, int] = {}
            
            # Get tasks from the last N days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Note: The tasks.list() method is not available in the current API
            # This method is kept for future API updates
            return {
                "status_counts": status_counts,
                "cutoff_date": cutoff_date.isoformat(),
            }
        except Exception as e:
            self.logger.error(f"Error checking usage: {str(e)}")
            raise APIError(f"Failed to check usage: {str(e)}")
