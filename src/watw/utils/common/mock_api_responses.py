"""
Mock API responses for testing.

This module provides mock responses for various APIs used in the application,
allowing for testing and development without consuming actual API credits.
"""

import os
import uuid
import json
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import cv2

# Check if OpenCV is available
OPENCV_AVAILABLE = True
try:
    import cv2
except ImportError:
    OPENCV_AVAILABLE = False

class MockResponses:
    """Class for generating mock API responses."""
    
    @staticmethod
    def get_tensorart_job_response(job_id: str) -> Dict[str, Any]:
        """
        Get a mock TensorArt job response.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict[str, Any]: Mock job response
        """
        return {
            "jobId": job_id,
            "status": "CREATED"
        }
    
    @staticmethod
    def get_tensorart_job_status_response(job_id: str, status: str = "SUCCEEDED", include_image: bool = True) -> Dict[str, Any]:
        """
        Get a mock TensorArt job status response.
        
        Args:
            job_id: Job ID
            status: Job status
            include_image: Whether to include image in output
            
        Returns:
            Dict[str, Any]: Mock job status response
        """
        response = {
            "job": {
                "id": job_id,
                "status": status,
                "createdAt": "2025-04-09T16:29:05.271758"
            }
        }
        
        if include_image:
            response["job"]["successInfo"] = {
                "images": [
                    {
                        "url": "/Users/dev/womanareoundtheworld/tests/mockdata/generated_clips/mock-image.png",
                        "width": 768,
                        "height": 1280
                    }
                ]
            }
        
        return response
    
    @staticmethod
    def get_runway_task_response(task_id: str) -> Dict[str, Any]:
        """
        Get a mock RunwayML task response.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Mock task response
        """
        return {
            "id": task_id,
            "status": "CREATED"
        }
    
    @staticmethod
    def get_runway_task_status_response(task_id: str, status: str = "COMPLETED", include_video: bool = True) -> Dict[str, Any]:
        """
        Get a mock RunwayML task status response.
        
        Args:
            task_id: Task ID
            status: Task status
            include_video: Whether to include video in output
            
        Returns:
            Dict[str, Any]: Mock task status response
        """
        response = {
            "id": task_id,
            "status": status
        }
        
        if include_video:
            response["output"] = [
                {
                    "url": "/Users/dev/womanareoundtheworld/tests/mockdata/generated_clips/mock-video.mp4",
                    "type": "video/mp4"
                }
            ]
        
        return response
    
    @staticmethod
    def get_rate_limit_response(retry_after: int) -> Dict[str, Any]:
        """
        Get a mock rate limit response.
        
        Args:
            retry_after: Seconds to wait before retrying
            
        Returns:
            Dict[str, Any]: Mock rate limit response
        """
        return {
            "error": f"Rate limit exceeded. Retry after {retry_after} seconds",
            "retry_after": retry_after
        }
    
    @staticmethod
    def get_runway_error_response(error_message: str) -> Dict[str, Any]:
        """
        Get a mock RunwayML error response.
        
        Args:
            error_message: Error message
            
        Returns:
            Dict[str, Any]: Mock error response
        """
        return {
            "error": error_message
        }
    
    @staticmethod
    def create_mock_image(width: int = 768, height: int = 1280, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Create a mock image for testing.
        
        Args:
            width: Image width
            height: Image height
            output_path: Output path for the image
            
        Returns:
            str: Path to the created image
        """
        # If no output path provided, create one
        if output_path is None:
            output_path = Path.cwd() / f"mock_image_{uuid.uuid4().hex[:8]}.png"
        else:
            output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if OPENCV_AVAILABLE:
            # Create a random image
            image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            cv2.imwrite(str(output_path), image)
        else:
            # Create a simple text file as a fallback
            with open(output_path, 'w') as f:
                f.write("Mock image data")
        
        return str(output_path)
    
    @staticmethod
    def create_mock_video(duration: int = 5, fps: int = 30, width: int = 768, height: int = 1280, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Create a mock video for testing.
        
        Args:
            duration: Video duration in seconds
            fps: Frames per second
            width: Video width
            height: Video height
            output_path: Output path for the video
            
        Returns:
            str: Path to the created video
        """
        # If no output path provided, create one
        if output_path is None:
            output_path = Path.cwd() / f"mock_video_{uuid.uuid4().hex[:8]}.mp4"
        else:
            output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if OPENCV_AVAILABLE:
            # Create a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Generate frames
            num_frames = duration * fps
            for _ in range(num_frames):
                # Create a random frame
                frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                out.write(frame)
            
            out.release()
        else:
            # Create a simple text file as a fallback
            with open(output_path, 'w') as f:
                f.write("Mock video data")
        
        return str(output_path)

# Create a singleton instance
mock_responses = MockResponses() 