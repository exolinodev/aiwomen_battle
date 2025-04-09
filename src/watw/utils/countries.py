"""
Country configurations for Women Around The World.
This module provides functions for accessing country data from the configuration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from watw.utils.common.exceptions import FileOperationError, ValidationError
from .prompts import BASE_IMAGE_TEMPLATE, ANIMATION_TEMPLATE, NEGATIVE_PROMPT

def load_config(config_path: str = 'config.json') -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        dict: Configuration data
        
    Raises:
        FileNotFoundError: If the config file cannot be found
        json.JSONDecodeError: If the config file is not valid JSON
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def get_country_info(country_code: Optional[str] = None, config_path: str = 'config.json') -> Dict[str, Any]:
    """Get country information by country code.
    
    Args:
        country_code: The country code to look up (e.g., 'US', 'brazil'). If None, returns all countries.
        config_path: Path to the configuration file
        
    Returns:
        dict: Country information including name, colors, flag description, and prompts.
              If country_code is None, returns a dictionary of all countries.
        
    Raises:
        ValueError: If the country code is provided but not found in the configuration
    """
    # Load configuration
    config = load_config(config_path)
    
    # If no country code provided, return all countries
    if country_code is None:
        return {k: v for k, v in config.items() if isinstance(v, dict) and 'id' in v}
    
    # Convert country code to lowercase for case-insensitive matching
    country_code = country_code.lower()
    
    # Find country data
    for key, value in config.items():
        if isinstance(value, dict) and key.lower() == country_code:
            return value
    
    raise ValueError(f"Country code '{country_code}' not found in configuration")

def format_prompt(template: str, country_data: Dict) -> str:
    """
    Format a prompt template with country data.
    
    Args:
        template: The prompt template string
        country_data: Dictionary containing country information
        
    Returns:
        str: Formatted prompt string
    """
    return template.format(
        colors=country_data.get('colors', ''),
        flag_description=country_data.get('flag_description', ''),
        name=country_data.get('name', '')
    )

@dataclass
class Country:
    """
    Represents a country with its configuration and prompts.
    
    Attributes:
        id: Unique identifier for the country
        name: Name of the country
        colors: Colors associated with the country's flag
        flag_description: Description of the country's flag
        base_image_prompt: Prompt for generating base images
        animation_prompt: Prompt for generating animations
        intro_voiceover: Optional voiceover for introduction
        transition_voiceover: Optional voiceover for transition
    """
    id: str
    name: str
    colors: str
    flag_description: str
    base_image_prompt: str
    animation_prompt: str
    intro_voiceover: Optional[str] = None
    transition_voiceover: Optional[str] = None

# Load countries and generate prompts
COUNTRIES = {}
config_data = load_config()

# Filter out non-country entries (like API keys)
for country_key, country_data in config_data.items():
    # Skip non-dictionary entries (like API keys)
    if not isinstance(country_data, dict):
        continue
        
    # Create a copy of the country data
    country_config = {**country_data}
    
    # Only use templates if the fields don't already exist in the config
    if 'base_image_prompt' not in country_data:
        country_config['base_image_prompt'] = format_prompt(BASE_IMAGE_TEMPLATE, country_data)
    
    if 'animation_prompt' not in country_data:
        country_config['animation_prompt'] = format_prompt(ANIMATION_TEMPLATE, country_data)
    
    COUNTRIES[country_key] = country_config

# Common negative prompts
BASE_IMAGE_NEGATIVE_PROMPT = NEGATIVE_PROMPT
ANIMATION_NEGATIVE_PROMPT = NEGATIVE_PROMPT 