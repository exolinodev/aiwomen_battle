"""
Credential management utilities for the Women Around The World project.

This module provides a centralized system for managing API credentials and other
sensitive configuration values. It handles loading credentials from environment
variables, validating them, and providing access to them throughout the application.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dotenv import load_dotenv
from watw.utils.common.exceptions import ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)

class CredentialManager:
    """
    Manages API credentials and other sensitive configuration values.
    
    This class provides a centralized way to access credentials throughout the
    application, with proper validation and error handling.
    """
    
    # Define required credentials with their environment variable names
    REQUIRED_CREDENTIALS = {
        "tensorart_bearer_token": "TENSORART_BEARER_TOKEN",
        "runwayml_api_secret": "RUNWAYML_API_SECRET"
    }
    
    # Define optional credentials with their environment variable names
    OPTIONAL_CREDENTIALS = {
        "elevenlabs_api_key": "ELEVENLABS_API_KEY",
        "openai_api_key": "OPENAI_API_KEY"
    }
    
    # Singleton instance
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CredentialManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the credential manager."""
        if self._initialized:
            return
            
        self._credentials: Dict[str, str] = {}
        self._initialized = True
        self._load_credentials()
    
    def _load_credentials(self) -> None:
        """
        Load credentials from environment variables.
        
        This method:
        1. Loads environment variables from .env file
        2. Validates required credentials
        3. Stores all credentials in the internal dictionary
        
        Raises:
            ConfigurationError: If required credentials are missing
        """
        # Load environment variables from .env file
        env_path = Path(os.getcwd()) / ".env"
        if env_path.exists():
            logger.info(f"Loading credentials from {env_path}")
            load_dotenv(env_path)
        else:
            logger.warning(f"No .env file found at {env_path}, using environment variables")
        
        # Check for required credentials
        missing_credentials = []
        for credential_name, env_var in self.REQUIRED_CREDENTIALS.items():
            value = os.getenv(env_var)
            if not value:
                missing_credentials.append(credential_name)
            else:
                self._credentials[credential_name] = value
        
        # Add optional credentials if they exist
        for credential_name, env_var in self.OPTIONAL_CREDENTIALS.items():
            value = os.getenv(env_var)
            if value:
                self._credentials[credential_name] = value
        
        # Raise error if any required credentials are missing
        if missing_credentials:
            raise ConfigurationError(
                f"Missing required credentials: {', '.join(missing_credentials)}. "
                "Please set them in your .env file."
            )
        
        logger.info("Credentials loaded successfully")
    
    def get(self, credential_name: str, default: Any = None) -> Optional[str]:
        """
        Get a credential by name.
        
        Args:
            credential_name: Name of the credential to retrieve
            default: Default value to return if credential is not found
            
        Returns:
            The credential value or the default if not found
        """
        return self._credentials.get(credential_name, default)
    
    def get_tensorart_token(self) -> str:
        """
        Get the TensorArt bearer token.
        
        Returns:
            The TensorArt bearer token
            
        Raises:
            ConfigurationError: If the token is not available
        """
        token = self.get("tensorart_bearer_token")
        if not token:
            raise ConfigurationError("TensorArt bearer token not found")
        return token
    
    def get_runwayml_secret(self) -> str:
        """
        Get the RunwayML API secret.
        
        Returns:
            The RunwayML API secret
            
        Raises:
            ConfigurationError: If the secret is not available
        """
        secret = self.get("runwayml_api_secret")
        if not secret:
            raise ConfigurationError("RunwayML API secret not found")
        return secret
    
    def get_elevenlabs_key(self) -> Optional[str]:
        """
        Get the ElevenLabs API key.
        
        Returns:
            The ElevenLabs API key or None if not available
        """
        return self.get("elevenlabs_api_key")
    
    def get_openai_key(self) -> Optional[str]:
        """
        Get the OpenAI API key.
        
        Returns:
            The OpenAI API key or None if not available
        """
        return self.get("openai_api_key")
    
    def reload(self) -> None:
        """
        Reload credentials from environment variables.
        
        This is useful when credentials are updated during runtime.
        
        Raises:
            ConfigurationError: If required credentials are missing after reload
        """
        self._credentials.clear()
        self._load_credentials()
    
    def is_available(self, credential_name: str) -> bool:
        """
        Check if a credential is available.
        
        Args:
            credential_name: Name of the credential to check
            
        Returns:
            True if the credential is available, False otherwise
        """
        return credential_name in self._credentials and bool(self._credentials[credential_name])


# Create a global instance for easy access
credentials = CredentialManager() 