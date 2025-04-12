from setuptools import setup, find_packages
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_readme() -> str:
    """
    Read the README.md file or return a default description.
    
    Returns:
        str: The contents of README.md or a default description
    """
    default_description = "Women Around The World - Video Generation Project"
    readme_path = "README.md"
    
    try:
        if not os.path.exists(readme_path):
            logger.warning("README.md not found, using default description")
            return default_description
            
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
            
    except Exception as e:
        logger.error(f"Error reading README.md: {e}")
        return default_description

# Setup configuration
setup(
    name="watw",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "click",
        "elevenlabs",
        "ffmpeg-python",
        "librosa",
        "moviepy",
        "numpy",
        "Pillow",
        "PyYAML",
        "python-dotenv",
        "requests",
        "rich",
        "runwayml",
        "scipy",
        "typing-extensions>=4.0.0",  # Required for ParamSpec in Python < 3.10
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'mypy',
            'ruff',  # Using ruff instead of flake8 and isort as it's faster and more modern
            'black',
        ]
    },
    python_requires=">=3.8",
    author="Your Name",
    author_email="your.email@example.com",
    description="Women Around The World - Video Generation Project",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 