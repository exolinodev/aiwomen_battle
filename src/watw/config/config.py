"""
Configuration management for Women Around The World.

This module provides a centralized configuration management system that handles
loading, saving, and accessing configuration settings from various sources such as
environment variables, configuration files, and default values.
"""

# Standard library imports
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Third-party imports
import yaml

# Define the central configuration file path
DEFAULT_CONFIG_FILE = (
    Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
    / "config"
    / "config.json"
)


class ConfigurationError(Exception):
    """
    Base exception class for configuration-related errors.

    This is the parent class for all configuration-specific exceptions.
    """

    pass


class ConfigurationFileError(ConfigurationError):
    """
    Exception raised for errors related to configuration files.

    This exception is raised when there are issues with reading from or writing to
    configuration files, such as file not found, permission denied, or invalid JSON.

    Attributes:
        file_path: Path to the configuration file that caused the error.
        message: A string describing the error.
    """

    def __init__(self, file_path: Union[str, Path], message: str):
        self.file_path = Path(file_path)
        self.message = message
        super().__init__(
            f"Configuration file error at {self.file_path}: {self.message}"
        )


class ConfigurationKeyError(ConfigurationError):
    """
    Exception raised when a configuration key is not found.

    This exception is raised when attempting to access a configuration key that
    doesn't exist in the configuration data.

    Attributes:
        key: The configuration key that was not found.
    """

    def __init__(self, key: str):
        self.key = key
        super().__init__(f"Configuration key not found: {self.key}")


class ConfigurationValueError(ConfigurationError):
    """
    Exception raised when a configuration value is invalid.

    This exception is raised when a configuration value doesn't meet the expected
    type or constraints.

    Attributes:
        key: The configuration key with the invalid value.
        message: A string describing why the value is invalid.
    """

    def __init__(self, key: str, message: str):
        self.key = key
        self.message = message
        super().__init__(
            f"Invalid value for configuration key '{self.key}': {self.message}"
        )


class Configuration:
    """
    A class to manage configuration settings for the Women Around The World application.

    This class provides a centralized way to manage configuration settings, with a
    hierarchy that prioritizes environment variables over configuration file values
    and default values in code. It supports loading and saving configurations to
    both JSON and YAML files, as well as accessing settings with type safety.

    Attributes:
        config_file (Path): Path to the configuration file.
        config_data (Dict[str, Any]): The current configuration data.
        default_config (Dict[str, Any]): Default configuration values.
    """

    def __init__(
        self,
        config_file: Union[str, Path],
        default_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Configuration object.

        Args:
            config_file: Path to the configuration file, either as a string or Path object.
            default_config: Default configuration values. If None, an empty dictionary is used.

        Examples:
            >>> config = Configuration("config.json")
            >>> config = Configuration("config.yaml")
            >>> config = Configuration("config.json", {"api_key": "default_key"})
        """
        self.config_file = Path(config_file)
        self.default_config = default_config or {}
        self.config_data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """
        Load configuration from the configuration file and environment variables.

        This method loads the configuration in the following order of precedence:
        1. Environment variables
        2. Configuration file values
        3. Default values

        If the configuration file doesn't exist, it will be created with default values.

        Raises:
            ConfigurationFileError: If there's an error reading the configuration file.

        Examples:
            >>> config = Configuration("config.json")
            >>> config.load()
        """
        # Load from file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                        self.config_data = yaml.safe_load(f) or {}
                    else:
                        self.config_data = json.load(f)
            except (json.JSONDecodeError, yaml.YAMLError) as e:
                raise ConfigurationFileError(
                    self.config_file, f"Invalid file format: {str(e)}"
                )
            except PermissionError:
                raise ConfigurationFileError(
                    self.config_file, "Permission denied when reading file"
                )
            except Exception as e:
                raise ConfigurationFileError(
                    self.config_file, f"Unexpected error: {str(e)}"
                )

        # Override with environment variables
        for key, value in os.environ.items():
            if key.startswith("WATW_"):
                config_key = key[5:].lower()
                self.config_data[config_key] = value

        # Add default values for missing keys
        for key, value in self.default_config.items():
            if key not in self.config_data:
                self.config_data[key] = value

        # Save the configuration to ensure all defaults are written
        self.save()

    def save(self) -> None:
        """
        Save the current configuration to the configuration file.

        This method writes the current configuration data to the specified
        configuration file in either JSON or YAML format based on the file extension.
        If the file doesn't exist, it will be created.

        Raises:
            ConfigurationFileError: If there's an error writing to the configuration file.

        Examples:
            >>> config = Configuration("config.json")
            >>> config.config_data["new_setting"] = "value"
            >>> config.save()
        """
        try:
            # Ensure the directory exists
            self.config_file.parent.mkdir(parents=True, exist_ok=True)

            # Write the configuration
            with open(self.config_file, "w") as f:
                if self.config_file.suffix.lower() in [".yaml", ".yml"]:
                    yaml.dump(self.config_data, f, default_flow_style=False)
                else:
                    json.dump(self.config_data, f, indent=4)
        except PermissionError:
            raise ConfigurationFileError(
                self.config_file, "Permission denied when writing file"
            )
        except Exception as e:
            raise ConfigurationFileError(
                self.config_file, f"Unexpected error: {str(e)}"
            )

    def get(self, key: str, default: Optional[Any] = None) -> Union[Any, None]:
        """
        Get a configuration value.

        This method retrieves a configuration value by key, returning the default
        value if the key doesn't exist.

        Args:
            key: The configuration key to retrieve.
            default: The default value to return if the key doesn't exist.

        Returns:
            The configuration value, or the default if the key doesn't exist.

        Examples:
            >>> config = Configuration("config.json")
            >>> api_key = config.get("api_key", "default_key")
            >>> debug_mode = config.get("debug", False)
        """
        return self.config_data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """
        Set a configuration value.

        This method sets a configuration value and saves the configuration to the file.

        Args:
            key: The configuration key to set.
            value: The value to set.

        Raises:
            ConfigurationValueError: If the value is invalid for the key.

        Examples:
            >>> config = Configuration("config.json")
            >>> config.set("api_key", "new_key")
            >>> config.set("debug", True)
        """
        # Validate the value if needed
        if key == "debug" and not isinstance(value, bool):
            raise ConfigurationValueError(key, "Debug mode must be a boolean value")

        self.config_data[key] = value
        self.save()

    def delete(self, key: str) -> None:
        """
        Delete a configuration value.

        This method removes a configuration value and saves the configuration to the file.

        Args:
            key: The configuration key to delete.

        Examples:
            >>> config = Configuration("config.json")
            >>> config.delete("api_key")
        """
        if key in self.config_data:
            del self.config_data[key]
            self.save()

    def __getitem__(self, key: str) -> Any:
        """
        Get a configuration value using dictionary-style access.

        This method allows accessing configuration values using dictionary-style
        syntax (e.g., config['api_key']).

        Args:
            key: The configuration key to retrieve.

        Returns:
            The configuration value.

        Raises:
            ConfigurationKeyError: If the key doesn't exist.

        Examples:
            >>> config = Configuration("config.json")
            >>> api_key = config['api_key']
        """
        if key not in self.config_data:
            raise ConfigurationKeyError(key)
        return self.config_data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set a configuration value using dictionary-style access.

        This method allows setting configuration values using dictionary-style
        syntax (e.g., config['api_key'] = 'new_key').

        Args:
            key: The configuration key to set.
            value: The value to set.

        Raises:
            ConfigurationValueError: If the value is invalid for the key.

        Examples:
            >>> config = Configuration("config.json")
            >>> config['api_key'] = 'new_key'
        """
        self.set(key, value)

    def __contains__(self, key: str) -> bool:
        """
        Check if a configuration key exists.

        This method allows checking if a configuration key exists using the 'in'
        operator (e.g., 'api_key' in config).

        Args:
            key: The configuration key to check.

        Returns:
            True if the key exists, False otherwise.

        Examples:
            >>> config = Configuration("config.json")
            >>> if 'api_key' in config:
            ...     print("API key exists")
        """
        return key in self.config_data

    def get_api_key(self, service: str) -> str:
        """
        Get API key for a service.

        This method retrieves an API key for a specific service, checking both
        environment variables and configuration file values.

        Args:
            service: Name of the service (e.g., 'elevenlabs', 'runwayml').

        Returns:
            The API key for the service.

        Raises:
            ConfigurationKeyError: If the API key is not found.

        Examples:
            >>> config = Configuration("config.json")
            >>> api_key = config.get_api_key('elevenlabs')
        """
        # Try environment variable first
        env_var = f"WATW_{service.upper()}_API_KEY"
        api_key = os.environ.get(env_var)

        # If not in environment, try config
        if not api_key:
            api_key = self.get(f"{service}_api_key")

        if not api_key:
            raise ConfigurationKeyError(f"API key for {service} not found")

        return api_key

    def get_base_url(self, service: str) -> str:
        """
        Get base URL for a service API.

        This method retrieves the base URL for a specific service API, checking both
        environment variables and configuration file values.

        Args:
            service: Name of the service (e.g., 'elevenlabs', 'runwayml').

        Returns:
            The base URL for the service API.

        Raises:
            ConfigurationKeyError: If the base URL is not found.

        Examples:
            >>> config = Configuration("config.json")
            >>> base_url = config.get_base_url('elevenlabs')
        """
        # Try environment variable first
        env_var = f"WATW_{service.upper()}_BASE_URL"
        base_url = os.environ.get(env_var)

        # If not in environment, try config
        if not base_url:
            base_url = self.get(f"{service}_base_url")

        if not base_url:
            raise ConfigurationKeyError(f"Base URL for {service} not found")

        return base_url
