"""
Women Around The World - Video Generation Project
"""

# Version information
__version__ = "0.1.0"

# Local imports
from watw.core.render import VideoRenderer
from watw.core.video_editor import VideoEditor
from watw.core.voiceover import VoiceoverGenerator
from watw.utils.countries import get_country_info
from watw.utils.prompts import generate_prompts

__all__ = [
    "VideoEditor",
    "VoiceoverGenerator",
    "VideoRenderer",
    "get_country_info",
    "generate_prompts",
]
