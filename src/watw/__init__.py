"""
Women Around The World - Video Generation Project
"""

# Version information
__version__ = "0.1.0"

from watw.core.Generation.countries import country_manager

# Local imports
from watw.core.render import VideoRenderer
from watw.core.video.base import VideoEditor
from watw.core.voiceover import VoiceoverGenerator

__all__ = [
    "VideoEditor",
    "VoiceoverGenerator",
    "VideoRenderer",
    "country_manager",
]
