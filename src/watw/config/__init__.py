"""
Configuration management for Women Around The World
"""

# Standard library imports
from pathlib import Path
from typing import Optional, Union

# Local imports
from watw.config.config import Configuration, DEFAULT_CONFIG_FILE


def load_config(config_path: Optional[Union[str, Path]] = None) -> Configuration:
    """
    Load configuration from config file and environment variables.

    Args:
        config_path (Union[str, Path], optional): Path to config.json file.
            Defaults to config/config.json in package root.

    Returns:
        Configuration: Configuration object
    """
    effective_config_path: Union[str, Path] = config_path if config_path is not None else DEFAULT_CONFIG_FILE
    config = Configuration(config_file=effective_config_path)
    return config


__all__ = ["Configuration", "load_config"]
