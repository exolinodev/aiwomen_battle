"""
Configuration management for Women Around The World
"""

# Local imports
from watw.config.config import Configuration

def load_config(config_path=None):
    """
    Load configuration from config file and environment variables.
    
    Args:
        config_path (str, optional): Path to config.json file. 
            Defaults to config/config.json in package root.
    
    Returns:
        Configuration: Configuration object
    """
    return Configuration.from_file(config_path)

__all__ = ["Configuration", "load_config"]
