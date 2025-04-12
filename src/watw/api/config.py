"""
API configuration for Women Around The World project.

This module provides functions for accessing API configurations from
the centralized configuration system.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from watw.config.config import Configuration
from watw.utils.common.exceptions import ConfigurationError

# Default configuration file path
CONFIG_PATH = (
    Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    / "config"
    / "config.json"
)

# Load configuration
config = Configuration(CONFIG_PATH)

# API key name mappings
API_KEY_MAPPINGS = {
    "elevenlabs": "elevenlabs_api_key",
    "runwayml": "runway_api_key",
    "tensorart": "tensorart_api_key",
}


class Config:
    """Configuration manager for API settings."""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.

        Args:
            config_file: Path to the configuration file
        """
        if config_file is None:
            config_file = os.path.join("config", "config.json")

        self.config_file = config_file
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Dict[str, Any]: Configuration data
        """
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {str(e)}")

    def get_setting(
        self, service: str, key: str, default: Optional[Any] = None
    ) -> Any:
        """
        Get a configuration setting.

        Args:
            service: Service name (e.g., 'tensorart', 'runwayml')
            key: Setting key
            default: Default value if not found

        Returns:
            Any: Setting value
        """
        try:
            if "api" in self.config and service in self.config["api"]:
                return self.config["api"][service].get(key, default)
            return default
        except Exception as e:
            raise ConfigurationError(f"Failed to get setting {key}: {str(e)}")

    def get_api_key(self, service: str) -> Optional[str]:
        """
        Get API key for a service.

        Args:
            service: Service name (e.g., 'tensorart', 'runwayml')

        Returns:
            Optional[str]: API key if found
        """
        try:
            if "api" in self.config and service in self.config["api"]:
                return self.config["api"][service].get("api_key")
            return None
        except Exception as e:
            raise ConfigurationError(f"Failed to get API key: {str(e)}")

    def get_base_url(self, service: str) -> Optional[str]:
        """
        Get base URL for a service.

        Args:
            service: Service name (e.g., 'tensorart', 'runwayml')

        Returns:
            Optional[str]: Base URL if found
        """
        try:
            if "api" in self.config and service in self.config["api"]:
                return self.config["api"][service].get("base_url")
            return None
        except Exception as e:
            raise ConfigurationError(f"Failed to get base URL: {str(e)}")

    def get_endpoint(self, service: str, endpoint: str) -> str:
        """
        Get endpoint URL for a service.

        Args:
            service: Name of the service
            endpoint: Endpoint name

        Returns:
            str: Endpoint URL
        """
        base_url = self.get_base_url(service)
        if base_url:
            return f"{base_url}/{endpoint}"
        else:
            raise ConfigurationError(f"Base URL for {service} not found")

    def get_model_id(self, service: str) -> Optional[str]:
        """
        Get model ID for a service.

        Args:
            service: Name of the service

        Returns:
            Optional[str]: Model ID or None if not found
        """
        model_id = self.get_setting(service, "model_id", None)
        if model_id is None:
            return None
        if not isinstance(model_id, str):
            return None
        return model_id if model_id else None


def get_api_key(service: str) -> str:
    """
    Get API key for a service.

    Args:
        service: Name of the service ('elevenlabs', 'runwayml', etc.)

    Returns:
        str: API key for the service

    Raises:
        ValueError: If API key is not found or is not a string
    """
    # Get from config using the mapped key name
    config_key = API_KEY_MAPPINGS.get(service, f"{service}_api_key")
    api_key = config.get_api_key(service)

    if not api_key:
        raise ValueError(
            f"API key for {service} not found. Set '{config_key}' in config.json"
        )
    if not isinstance(api_key, str):
        raise ValueError(
            f"API key for {service} must be a string, got {type(api_key)}"
        )
    return api_key


def get_base_url(service: str) -> str:
    """
    Get base URL for a service.

    Args:
        service: Name of the service

    Returns:
        str: Base URL for the service

    Raises:
        ValueError: If base URL is not a string
    """
    base_url = config.get_base_url(service)
    if not isinstance(base_url, str):
        raise ValueError(
            f"Base URL for {service} must be a string, got {type(base_url)}"
        )
    return base_url


def get_model_id(service: str) -> Optional[str]:
    """
    Get model ID for a service.

    Args:
        service: Name of the service

    Returns:
        Optional[str]: Model ID or None if not found
    """
    model_id = config.get(f"{service}_model_id", None)
    if model_id is None:
        return None
    if not isinstance(model_id, str):
        return None
    return model_id if model_id else None
