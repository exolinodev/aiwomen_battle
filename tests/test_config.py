"""
Test configuration for setting up mock data paths and test environment.
"""

import os
from pathlib import Path

# Set up mock data paths
MOCK_DATA_DIR = Path("/Users/dev/womanareoundtheworld/tests/mockdata")
MOCK_VOICEOVERS_DIR = MOCK_DATA_DIR / "voiceovers"
MOCK_GENERATED_CLIPS_DIR = MOCK_DATA_DIR / "generated_clips"
MOCK_TEMP_FILES_DIR = MOCK_DATA_DIR / "temp_files"
MOCK_FINAL_VIDEO_DIR = MOCK_DATA_DIR / "final_video"

# Ensure mock directories exist
for directory in [MOCK_VOICEOVERS_DIR, MOCK_GENERATED_CLIPS_DIR, MOCK_TEMP_FILES_DIR, MOCK_FINAL_VIDEO_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Test configuration
TEST_CONFIG = {
    "mock_data_dir": str(MOCK_DATA_DIR),
    "mock_voiceovers_dir": str(MOCK_VOICEOVERS_DIR),
    "mock_generated_clips_dir": str(MOCK_GENERATED_CLIPS_DIR),
    "mock_temp_files_dir": str(MOCK_TEMP_FILES_DIR),
    "mock_final_video_dir": str(MOCK_FINAL_VIDEO_DIR),
    
    # Mock API keys for testing
    "runwayml": {
        "api_key": "mock_key_runwayml",
        "base_url": "https://mock-runwayml-api.example.com"
    },
    "tensorart": {
        "api_key": "mock_key_tensorart",
        "base_url": "https://mock-tensorart-api.example.com",
        "submit_job_endpoint": "/v1/jobs",
        "model_id": "mock_model_id"
    },
    "elevenlabs": {
        "api_key": "mock_key_elevenlabs",
        "voice": "mock_voice",
        "model": "mock_model"
    }
} 