"""
Video editing module for Women Around The World.

This module provides classes for video editing and processing, including:
- Base video editor functionality
- Rhythmic video editing with beat detection
- Enhanced video editing with transitions and effects
"""

from watw.core.video.base import VideoEditor
from watw.core.video.enhanced import EnhancedRhythmicVideoEditor
from watw.core.video.rhythmic import BeatInfo, RhythmicVideoEditor

__all__ = [
    "VideoEditor",
    "RhythmicVideoEditor",
    "EnhancedRhythmicVideoEditor",
    "BeatInfo",
]
