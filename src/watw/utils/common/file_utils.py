"""
File utility functions for Women Around The World
"""

# Standard library imports
import shutil
from pathlib import Path
from typing import Optional, Union


def ensure_directory(directory: Union[str, Path], clean: bool = False) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory
        clean: If True, clean the directory if it exists

    Returns:
        Path object for the directory
    """
    directory_path = Path(directory)

    if clean and directory_path.exists():
        shutil.rmtree(directory_path)

    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path


def get_file_extension(file_path: Union[str, Path]) -> str:
    """
    Get the file extension from a file path.

    Args:
        file_path: Path to the file

    Returns:
        File extension (including the dot)
    """
    return Path(file_path).suffix


def create_temp_file(
    content: str, extension: str = ".txt", directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a temporary file with the given content.

    Args:
        content: Content to write to the file
        extension: File extension (including the dot)
        directory: Directory to create the file in (defaults to system temp directory)

    Returns:
        Path to the created file
    """
    import tempfile

    if directory is None:
        directory = tempfile.gettempdir()

    directory_path = ensure_directory(directory)

    # Create a temporary file with the given extension
    temp_file = tempfile.NamedTemporaryFile(
        suffix=extension, dir=directory_path, delete=False
    )
    temp_file.write(content.encode())
    temp_file.close()

    return Path(temp_file.name)


def copy_file(source: Union[str, Path], destination: Union[str, Path]) -> Path:
    """
    Copy a file from source to destination.

    Args:
        source: Source file path
        destination: Destination file path

    Returns:
        Path to the destination file
    """
    source_path = Path(source)
    destination_path = Path(destination)

    # Ensure the destination directory exists
    ensure_directory(destination_path.parent)

    # Copy the file
    shutil.copy2(source_path, destination_path)

    return destination_path
