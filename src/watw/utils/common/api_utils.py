"""
API utility functions for Women Around The World.

This module provides utility functions for interacting with various APIs,
including RunwayML, TensorArt, and ElevenLabs. It centralizes API-related
functionality to reduce code duplication and improve maintainability.
"""

# Standard library imports
import os
import json
import time
import base64
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
from enum import Enum

# Third-party imports
from dotenv import load_dotenv
from runwayml import RunwayML

# Local imports
from watw.utils.common.validation_utils import validate_api_key, ValidationError
from watw.utils.common.logging_utils import setup_logger

# Set up logger
logger = setup_logger("watw.api")

# Get workspace root directory
WORKSPACE_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.json"

class TaskStatus(Enum):
    """
    Enum representing the status of an API task.
    
    Attributes:
        PENDING: Task is waiting to be processed
        PROCESSING: Task is currently being processed
        SUCCEEDED: Task has completed successfully
        FAILED: Task has failed
    """
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"

class APIError(Exception):
    """
    Base exception class for API-related errors.
    
    This is the parent class for all API-specific exceptions.
    """
    pass

class APIResponseError(APIError):
    """
    Exception raised when an API returns an error response.
    
    Attributes:
        service: Name of the API service
        status_code: HTTP status code
        response: Response content
        message: Error message
    """
    def __init__(self, service: str, status_code: int, response: Any, message: Optional[str] = None):
        self.service = service
        self.status_code = status_code
        self.response = response
        self.message = message or f"{service} API returned status code {status_code}"
        super().__init__(self.message)

class APITimeoutError(APIError):
    """
    Exception raised when an API request times out.
    
    Attributes:
        service: Name of the API service
        timeout: Timeout value in seconds
    """
    def __init__(self, service: str, timeout: int):
        self.service = service
        self.timeout = timeout
        super().__init__(f"{service} API request timed out after {timeout} seconds")

class APIConfiguration:
    """
    Configuration for API services.
    
    This class manages API configuration settings, including API keys,
    endpoints, and other service-specific parameters.
    
    Attributes:
        config_file: Path to the configuration file
        config_data: Configuration data
    """
    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """
        Initialize the API configuration.
        
        Args:
            config_file: Path to the configuration file. If None, uses default location.
        """
        # Load environment variables
        load_dotenv()
        
        # Set default configuration
        self.config_data = {
            "runwayml": {
                "api_key": None,
                "base_url": "https://api.runwayml.com",
            },
            "tensorart": {
                "api_key": None,
                "base_url": "https://ap-east-1.tensorart.cloud",
                "submit_job_endpoint": "/v1/jobs",
                "model_id": "757279507095956705",  # Default model ID
            },
            "elevenlabs": {
                "api_key": None,
                "voice": "Bella",
                "model": "eleven_multilingual_v2",
            }
        }
        
        # Load from config file if provided, otherwise use default
        config_path = Path(config_file) if config_file else DEFAULT_CONFIG_PATH
        logger.info(f"Loading configuration from: {config_path}")
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                # Update with file config
                for service, settings in file_config.items():
                    if service in self.config_data:
                        if isinstance(settings, dict):
                            self.config_data[service].update(settings)
                        else:
                            # Handle case where settings is just an API key
                            self.config_data[service]["api_key"] = settings
            logger.info("Configuration loaded successfully")
        else:
            logger.warning(f"Configuration file not found at: {config_path}")
        
        # Fall back to environment variables if config file values are None
        if not self.config_data["runwayml"]["api_key"]:
            self.config_data["runwayml"]["api_key"] = os.getenv("RUNWAYML_API_SECRET")
        if not self.config_data["tensorart"]["api_key"]:
            self.config_data["tensorart"]["api_key"] = os.getenv("TENSORART_BEARER_TOKEN")
        if not self.config_data["elevenlabs"]["api_key"]:
            self.config_data["elevenlabs"]["api_key"] = os.getenv("ELEVENLABS_API_SECRET")
    
    def get_api_key(self, service: str) -> str:
        """
        Get the API key for a service.
        
        Args:
            service: Name of the API service
            
        Returns:
            API key
            
        Raises:
            ValidationError: If the API key is missing
        """
        api_key = self.config_data.get(service, {}).get("api_key")
        validate_api_key(api_key, service)
        return api_key
    
    def get_base_url(self, service: str) -> str:
        """
        Get the base URL for a service.
        
        Args:
            service: Name of the API service
            
        Returns:
            Base URL
        """
        return self.config_data.get(service, {}).get("base_url", "")
    
    def get_endpoint(self, service: str, endpoint: str) -> str:
        """
        Get a full endpoint URL for a service.
        
        Args:
            service: Name of the API service
            endpoint: Endpoint path
            
        Returns:
            Full endpoint URL
        """
        base_url = self.get_base_url(service)
        return f"{base_url}{endpoint}"
    
    def get_model_id(self, service: str) -> str:
        """
        Get the model ID for a service.
        
        Args:
            service: Name of the API service
            
        Returns:
            Model ID
        """
        return self.config_data.get(service, {}).get("model_id", "")

# Create a global instance
api_config = APIConfiguration()

def check_api_credentials(services: Optional[List[str]] = None) -> Dict[str, bool]:
    """
    Check if API credentials are available for the specified services.
    
    Args:
        services: List of services to check. If None, checks all configured services.
        
    Returns:
        Dictionary mapping service names to boolean values indicating if credentials are available
    """
    if services is None:
        services = list(api_config.config_data.keys())
    
    results = {}
    for service in services:
        try:
            api_config.get_api_key(service)
            results[service] = True
        except ValidationError:
            results[service] = False
    
    return results

def download_file(url: str, save_path: Union[str, Path]) -> Path:
    """
    Download a file from a URL and save it to the specified path.
    
    Args:
        url: URL to download from
        save_path: Path to save the file to
        
    Returns:
        Path to the saved file
        
    Raises:
        APIError: If the download fails
    """
    save_path = Path(save_path)
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path
    except requests.exceptions.RequestException as e:
        raise APIError(f"Failed to download file from {url}: {str(e)}")

def encode_image_base64(image_path: Union[str, Path]) -> str:
    """
    Encode an image file as a base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64-encoded image string
        
    Raises:
        APIError: If the encoding fails
    """
    image_path = Path(image_path)
    
    try:
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        raise APIError(f"Failed to encode image {image_path}: {str(e)}")

def poll_api_task(
    service: str,
    task_id: str,
    check_status_func: callable,
    timeout: int = 300,
    interval: int = 5,
    success_status: Union[str, List[str]] = TaskStatus.SUCCEEDED.value,
    failure_status: Union[str, List[str]] = TaskStatus.FAILED.value
) -> Dict[str, Any]:
    """
    Poll an API task until it completes or times out.
    
    Args:
        service: Name of the API service
        task_id: ID of the task to poll
        check_status_func: Function to check the task status
        timeout: Maximum time to wait in seconds
        interval: Time between polls in seconds
        success_status: Status value(s) indicating success
        failure_status: Status value(s) indicating failure
        
    Returns:
        Task result
        
    Raises:
        APITimeoutError: If the task times out
        APIError: If the task fails
    """
    start_time = time.time()
    
    # Convert to lists if needed
    if isinstance(success_status, str):
        success_status = [success_status]
    if isinstance(failure_status, str):
        failure_status = [failure_status]
    
    while True:
        # Check if timeout has been reached
        if time.time() - start_time > timeout:
            raise APITimeoutError(service, timeout)
        
        # Check task status
        result = check_status_func(task_id)
        status = result.get("status")
        
        # Check if task has completed
        if status in success_status:
            return result
        elif status in failure_status:
            raise APIError(f"{service} task {task_id} failed: {result.get('error', 'Unknown error')}")
        
        # Wait before polling again
        time.sleep(interval)

def get_runway_client() -> RunwayML:
    """
    Get a RunwayML client with the configured API key.
    
    Returns:
        RunwayML client
        
    Raises:
        ValidationError: If the API key is missing
    """
    api_key = api_config.get_api_key("runwayml")
    return RunwayML(api_key=api_key) 