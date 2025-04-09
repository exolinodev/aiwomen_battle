"""
Validation utility functions for Women Around The World.

This module provides utility functions for validating various inputs such as
files, directories, API keys, and data structures.
"""

# Standard library imports
import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Any

class ValidationError(Exception):
    """
    Exception raised for validation errors.
    
    This exception is raised when a validation check fails, such as when a file
    doesn't exist or an API key is missing.
    
    Attributes:
        message: A string describing the validation error.
    """
    pass

def validate_file_exists(file_path: Union[str, Path], error_message: Optional[str] = None) -> Path:
    """
    Validate that a file exists.
    
    This function checks if the specified file exists and returns a Path object
    if it does. If the file doesn't exist, it raises a ValidationError.
    
    Args:
        file_path: Path to the file, either as a string or Path object.
        error_message: Custom error message to use if the file doesn't exist.
                      If None, a default message will be used.
        
    Returns:
        Path: A Path object representing the file.
        
    Raises:
        ValidationError: If the file doesn't exist.
        
    Examples:
        >>> validate_file_exists("config.json")
        Path('config.json')
        >>> validate_file_exists("nonexistent.txt")
        ValidationError: File does not exist: nonexistent.txt
    """
    file_path = Path(file_path)
    
    if not file_path.is_file():
        if error_message is None:
            error_message = f"File does not exist: {file_path}"
        raise ValidationError(error_message)
    
    return file_path

def validate_directory_exists(directory: Union[str, Path], error_message: Optional[str] = None) -> Path:
    """
    Validate that a directory exists.
    
    This function checks if the specified directory exists and returns a Path object
    if it does. If the directory doesn't exist, it raises a ValidationError.
    
    Args:
        directory: Path to the directory, either as a string or Path object.
        error_message: Custom error message to use if the directory doesn't exist.
                      If None, a default message will be used.
        
    Returns:
        Path: A Path object representing the directory.
        
    Raises:
        ValidationError: If the directory doesn't exist.
        
    Examples:
        >>> validate_directory_exists("output")
        Path('output')
        >>> validate_directory_exists("nonexistent")
        ValidationError: Directory does not exist: nonexistent
    """
    directory_path = Path(directory)
    
    if not directory_path.is_dir():
        if error_message is None:
            error_message = f"Directory does not exist: {directory_path}"
        raise ValidationError(error_message)
    
    return directory_path

def validate_required_fields(data: Dict[str, Any], required_fields: List[str], error_message: Optional[str] = None) -> None:
    """
    Validate that all required fields are present in a dictionary.
    
    This function checks if all the required fields are present in the given dictionary.
    If any required field is missing, it raises a ValidationError.
    
    Args:
        data: The dictionary to validate.
        required_fields: List of field names that must be present in the dictionary.
        error_message: Custom error message to use if any required field is missing.
                      If None, a default message will be used.
        
    Raises:
        ValidationError: If any required field is missing.
        
    Examples:
        >>> data = {"name": "John", "age": 30}
        >>> validate_required_fields(data, ["name", "age"])
        >>> validate_required_fields(data, ["name", "email"])
        ValidationError: Missing required fields: email
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        if error_message is None:
            error_message = f"Missing required fields: {', '.join(missing_fields)}"
        raise ValidationError(error_message)

def validate_file_extension(file_path: Union[str, Path], allowed_extensions: List[str], error_message: Optional[str] = None) -> None:
    """
    Validate that a file has an allowed extension.
    
    This function checks if the file extension is in the list of allowed extensions.
    If the extension is not allowed, it raises a ValidationError.
    
    Args:
        file_path: Path to the file, either as a string or Path object.
        allowed_extensions: List of allowed extensions, including the dot (e.g., [".txt", ".json"]).
        error_message: Custom error message to use if the extension is not allowed.
                      If None, a default message will be used.
        
    Raises:
        ValidationError: If the file extension is not allowed.
        
    Examples:
        >>> validate_file_extension("document.txt", [".txt", ".md"])
        >>> validate_file_extension("image.jpg", [".png", ".gif"])
        ValidationError: File extension '.jpg' is not allowed. Allowed extensions: .png, .gif
    """
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    if extension not in allowed_extensions:
        if error_message is None:
            error_message = f"File extension '{extension}' is not allowed. Allowed extensions: {', '.join(allowed_extensions)}"
        raise ValidationError(error_message)

def validate_api_key(api_key: Optional[str], service_name: str, error_message: Optional[str] = None) -> None:
    """
    Validate that an API key is provided.
    
    This function checks if the API key is provided (not None or empty).
    If the API key is not provided, it raises a ValidationError.
    
    Args:
        api_key: The API key to validate.
        service_name: The name of the service (used in the error message).
        error_message: Custom error message to use if the API key is not provided.
                      If None, a default message will be used.
        
    Raises:
        ValidationError: If the API key is not provided.
        
    Examples:
        >>> validate_api_key("abc123", "ServiceName")
        >>> validate_api_key("", "ServiceName")
        ValidationError: ServiceName API key is required
        >>> validate_api_key(None, "ServiceName")
        ValidationError: ServiceName API key is required
    """
    if not api_key:
        if error_message is None:
            error_message = f"{service_name} API key is required"
        raise ValidationError(error_message) 