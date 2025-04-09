"""
Logging utility functions for Women Around The World
"""

# Standard library imports
import functools
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Any

def setup_logger(
    name: str = "watw",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the given name and level.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Path to log file (optional)
        format_string: Format string for log messages (optional)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_execution_time(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log the execution time of a function.
    
    Args:
        logger: Logger to use (optional)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            else:
                log = logger
            
            # Log start time
            start_time = time.time()
            log.debug(f"Starting {func.__name__}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log end time
            end_time = time.time()
            execution_time = end_time - start_time
            log.debug(f"Finished {func.__name__} in {execution_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator

def log_function_call(logger: Optional[logging.Logger] = None, level: int = logging.DEBUG) -> Callable:
    """
    Decorator to log function calls with arguments.
    
    Args:
        logger: Logger to use (optional)
        level: Logging level
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get logger
            if logger is None:
                log = logging.getLogger(func.__module__)
            else:
                log = logger
            
            # Log function call
            log.log(level, f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Log return value
            log.log(level, f"{func.__name__} returned: {result}")
            
            return result
        return wrapper
    return decorator 