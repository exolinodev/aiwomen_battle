"""
Country configurations for the swimsuit around the world generation workflow.
This module provides a structured way to manage country data and prompts.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
from watw.utils.common.exceptions import FileOperationError, ValidationError
from .prompts import BASE_IMAGE_TEMPLATE, ANIMATION_TEMPLATE, NEGATIVE_PROMPT

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
    
    @classmethod
    def from_dict(cls, country_id: str, data: Dict) -> 'Country':
        """
        Create a Country instance from a dictionary.
        
        Args:
            country_id: The country identifier
            data: Dictionary containing country data
            
        Returns:
            Country: A new Country instance
            
        Raises:
            ValidationError: If required fields are missing
        """
        required_fields = ['name', 'colors', 'flag_description']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            raise ValidationError(
                f"Country '{country_id}' is missing required fields: {', '.join(missing_fields)}"
            )
            
        # Use provided prompts or generate from templates
        base_image_prompt = data.get('base_image_prompt') or cls._format_prompt(
            BASE_IMAGE_TEMPLATE,
            data
        )
        
        animation_prompt = data.get('animation_prompt') or cls._format_prompt(
            ANIMATION_TEMPLATE,
            data
        )
        
        return cls(
            id=data.get('id', country_id),
            name=data['name'],
            colors=data['colors'],
            flag_description=data['flag_description'],
            base_image_prompt=base_image_prompt,
            animation_prompt=animation_prompt,
            intro_voiceover=data.get('intro_voiceover'),
            transition_voiceover=data.get('transition_voiceover')
        )
    
    @staticmethod
    def _format_prompt(template: str, data: Dict) -> str:
        """
        Format a prompt template with country data.
        
        Args:
            template: The prompt template to format
            data: Dictionary containing country data
            
        Returns:
            str: Formatted prompt
            
        Raises:
            ValidationError: If required fields are missing
        """
        try:
            return template.format(
                name=data['name'],
                colors=data['colors'],
                flag_description=data['flag_description']
            )
        except KeyError as e:
            raise ValidationError(f"Missing required field in country data: {str(e)}")
        except Exception as e:
            raise ValidationError(f"Failed to format prompt: {str(e)}")

class CountryManager:
    """
    Manages country configurations and provides access to country data.
    """
    
    def __init__(self):
        """Initialize the CountryManager with an empty dictionary of countries."""
        self._countries: Dict[str, Country] = {}
        
    def load_countries(self) -> None:
        """
        Load country configurations from config.json.
        
        Raises:
            FileOperationError: If the config file cannot be read or parsed
            ValidationError: If the config file is missing required fields
        """
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        
        try:
            if not os.path.exists(config_path):
                raise FileOperationError(f"Config file not found at: {config_path}")
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            # Convert dictionary data to Country objects
            for country_id, country_data in config_data.items():
                try:
                    self._countries[country_id] = Country.from_dict(country_id, country_data)
                except ValidationError as e:
                    raise ValidationError(f"Error loading country '{country_id}': {str(e)}")
                    
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Failed to parse config.json: {str(e)}")
        except Exception as e:
            raise FileOperationError(f"Unexpected error loading config.json: {str(e)}")
    
    def get_country(self, country_id: str) -> Optional[Country]:
        """
        Get a country by its ID.
        
        Args:
            country_id: The country identifier
            
        Returns:
            Optional[Country]: The country if found, None otherwise
        """
        return self._countries.get(country_id)
    
    def get_all_countries(self) -> List[Country]:
        """
        Get all countries.
        
        Returns:
            List[Country]: List of all countries
        """
        return list(self._countries.values())
    
    def get_country_by_index(self, index: int) -> Optional[Country]:
        """
        Get a country by its index in the list.
        
        Args:
            index: The index of the country
            
        Returns:
            Optional[Country]: The country if found, None otherwise
        """
        countries = self.get_all_countries()
        return countries[index] if 0 <= index < len(countries) else None

# Initialize the country manager and load countries
country_manager = CountryManager()
try:
    country_manager.load_countries()
except (FileOperationError, ValidationError) as e:
    print(f"Error initializing country configurations: {str(e)}")
    raise  # Re-raise to prevent usage with invalid configuration

# Common negative prompts
BASE_IMAGE_NEGATIVE_PROMPT: str = NEGATIVE_PROMPT
ANIMATION_NEGATIVE_PROMPT: str = NEGATIVE_PROMPT

# For backward compatibility
COUNTRIES = {country_id: country.__dict__ for country_id, country in country_manager._countries.items()} 