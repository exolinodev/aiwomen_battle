"""
Country configurations for the swimsuit around the world generation workflow.
This file loads country data from config.json and combines it with prompt templates.
"""
import os
import json
from .prompts import BASE_IMAGE_TEMPLATE, ANIMATION_TEMPLATE, NEGATIVE_PROMPT

def load_countries():
    """Load country configurations from config.json"""
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def format_prompt(template, country_data):
    """Format a prompt template with country data"""
    return template.format(
        name=country_data['name'],
        colors=country_data['colors'],
        flag_description=country_data['flag_description']
    )

# Load countries and generate prompts
COUNTRIES = {}
for country_key, country_data in load_countries().items():
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