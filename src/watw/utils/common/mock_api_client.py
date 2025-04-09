"""
Mock API clients for testing without using actual API credits.

This module provides mock clients for various APIs used in the application,
allowing for testing and development without consuming actual API credits.
"""

import time
import uuid
import json
from typing import Dict, Any, Optional, List, Union, Callable
from pathlib import Path
import requests
from requests.models import Response

from watw.utils.common.mock_api_responses import mock_responses
from watw.utils.common.exceptions import (
    TensorArtError,
    RunwayMLError,
    RateLimitExceeded,
    ValidationError
)

class MockTensorArtClient:
    """
    Mock client for TensorArt API.
    
    This class provides mock implementations of TensorArt API endpoints,
    allowing for testing and development without consuming actual API credits.
    """
    
    def __init__(self, api_key: str = "mock-api-key"):
        """
        Initialize the mock TensorArt client.
        
        Args:
            api_key: Mock API key
        """
        self.api_key = api_key
        self.base_url = "https://mock-tensorart-api.example.com"
        self.jobs = {}
        self._setup_mock_jobs()
        
    def _setup_mock_jobs(self):
        """Set up mock jobs for testing."""
        # Create a few mock jobs with different statuses
        self.jobs["mock-job-pending"] = {
            "id": "mock-job-pending",
            "status": "PENDING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        self.jobs["mock-job-processing"] = {
            "id": "mock-job-processing",
            "status": "PROCESSING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        self.jobs["mock-job-succeeded"] = {
            "id": "mock-job-succeeded",
            "status": "SUCCEEDED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "success_info": {
                "images": [
                    {
                        "url": "https://example.com/mock-image.png",
                        "width": 768,
                        "height": 1280
                    }
                ]
            }
        }
        
        self.jobs["mock-job-failed"] = {
            "id": "mock-job-failed",
            "status": "FAILED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "error": "Job failed due to invalid parameters"
        }
        
        self.jobs["mock-job-rate-limited"] = {
            "id": "mock-job-rate-limited",
            "status": "RATE_LIMITED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "retry_after": 60
        }
    
    def submit_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a mock job to TensorArt API.
        
        Args:
            payload: Job payload
            
        Returns:
            Dict[str, Any]: Mock job submission response
            
        Raises:
            TensorArtError: If job submission fails
            RateLimitExceeded: If rate limit is exceeded
        """
        # Simulate rate limiting
        if "rate_limit" in payload.get("request_id", ""):
            raise RateLimitExceeded("Rate limit exceeded", 60)
            
        # Simulate job submission failure
        if "fail" in payload.get("request_id", ""):
            raise TensorArtError("Job submission failed", status_code=400)
            
        # Generate a job ID
        job_id = f"mock-job-{uuid.uuid4().hex[:8]}"
        
        # Create a mock job
        job = {
            "id": job_id,
            "status": "CREATED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        # Store the job
        self.jobs[job_id] = job
        
        # Return the job submission response
        return mock_responses.get_tensorart_job_response(job_id)
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a mock job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dict[str, Any]: Mock job status response
            
        Raises:
            TensorArtError: If job status retrieval fails
            RateLimitExceeded: If rate limit is exceeded
        """
        # Check if the job exists
        if job_id not in self.jobs:
            raise TensorArtError(f"Job {job_id} not found", status_code=404)
            
        # Get the job
        job = self.jobs[job_id]
        
        # Simulate rate limiting
        if job["status"] == "RATE_LIMITED":
            raise RateLimitExceeded("Rate limit exceeded", job.get("retry_after", 60))
            
        # Simulate job status retrieval failure
        if "fail" in job_id:
            raise TensorArtError("Job status retrieval failed", status_code=500)
            
        # Update the job status based on the job ID
        if "pending" in job_id:
            job["status"] = "PENDING"
        elif "processing" in job_id:
            job["status"] = "PROCESSING"
        elif "succeeded" in job_id:
            job["status"] = "SUCCEEDED"
            if "success_info" not in job:
                job["success_info"] = {
                    "images": [
                        {
                            "url": "https://example.com/mock-image.png",
                            "width": 768,
                            "height": 1280
                        }
                    ]
                }
        elif "failed" in job_id:
            job["status"] = "FAILED"
            if "error" not in job:
                job["error"] = "Job failed due to invalid parameters"
                
        # Return the job status response
        return mock_responses.get_tensorart_job_status_response(
            job_id=job_id,
            status=job["status"],
            include_image="success_info" in job
        )

class MockRunwayMLClient:
    """
    Mock client for RunwayML API.
    
    This class provides mock implementations of RunwayML API endpoints,
    allowing for testing and development without consuming actual API credits.
    """
    
    def __init__(self, api_key: str = "mock-api-key"):
        """
        Initialize the mock RunwayML client.
        
        Args:
            api_key: Mock API key
            
        Raises:
            RunwayMLError: If API key is invalid
        """
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        # For testing purposes, allow both mock API key and key_ format
        if api_key != "mock-api-key" and not api_key.startswith("key_mock_"):
            raise RunwayMLError("The API key you provided is correctly formatted, but it doesn't represent an active key. Has it been deactivated?")
            
        self.api_key = api_key
        self.base_url = "https://mock-runwayml-api.example.com"
        self._tasks = {}
        self._setup_mock_tasks()
        
    def _setup_mock_tasks(self):
        """Set up mock tasks for testing."""
        # Create a few mock tasks with different statuses
        self._tasks["mock-task-pending"] = {
            "id": "mock-task-pending",
            "status": "PENDING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        self._tasks["mock-task-processing"] = {
            "id": "mock-task-processing",
            "status": "PROCESSING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        self._tasks["mock-task-completed"] = {
            "id": "mock-task-completed",
            "status": "COMPLETED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "output": {
                "url": "https://example.com/mock-video.mp4",
                "type": "video/mp4"
            }
        }
        
        self._tasks["mock-task-failed"] = {
            "id": "mock-task-failed",
            "status": "FAILED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "error": "Task failed due to invalid parameters"
        }
        
        self._tasks["mock-task-rate-limited"] = {
            "id": "mock-task-rate-limited",
            "status": "RATE_LIMITED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "retry_after": 60
        }
    
    def image_to_video(self):
        """Get the image to video API."""
        return MockImageToVideoAPI(self)
    
    @property
    def tasks(self):
        """Get the tasks API."""
        return MockTasksAPI(self)

class MockImageToVideoAPI:
    """Mock API for image to video conversion."""
    
    def __init__(self, client):
        """Initialize the mock image to video API."""
        self.client = client
    
    def create(self, **kwargs) -> Dict[str, Any]:
        """
        Create a mock image to video task.
        
        Args:
            **kwargs: Task parameters
            
        Returns:
            Dict[str, Any]: Mock task response
            
        Raises:
            RunwayMLError: If task creation fails
            RateLimitExceeded: If rate limit is exceeded
        """
        # Simulate rate limiting
        if "rate_limit" in kwargs.get("prompt_text", ""):
            raise RateLimitExceeded("Rate limit exceeded", 60)
            
        # Simulate task creation failure
        if "fail" in kwargs.get("prompt_text", ""):
            raise RunwayMLError("Task creation failed", status_code=400)
            
        # Generate a task ID
        task_id = f"mock-task-{uuid.uuid4().hex[:8]}"
        
        # Create a mock task
        task = {
            "id": task_id,
            "status": "CREATED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        # Store the task
        self.client._tasks[task_id] = task
        
        # Return the task creation response
        return mock_responses.get_runway_task_response(task_id)

class MockTasksAPI:
    """Mock API for task management."""
    
    def __init__(self, client):
        """Initialize the mock tasks API."""
        self.client = client
    
    def retrieve(self, task_id: str) -> Dict[str, Any]:
        """
        Retrieve a mock task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Mock task response
            
        Raises:
            RunwayMLError: If task retrieval fails
            RateLimitExceeded: If rate limit is exceeded
        """
        # Check if the task exists
        if task_id not in self.client._tasks:
            raise RunwayMLError(f"Task {task_id} not found", status_code=404)
            
        # Get the task
        task = self.client._tasks[task_id]
        
        # Simulate rate limiting
        if task["status"] == "RATE_LIMITED":
            raise RateLimitExceeded("Rate limit exceeded", task.get("retry_after", 60))
            
        # Simulate task retrieval failure
        if "fail" in task_id:
            raise RunwayMLError("Task retrieval failed", status_code=500)
            
        # Update the task status based on the task ID
        if "pending" in task_id:
            task["status"] = "PENDING"
        elif "processing" in task_id:
            task["status"] = "PROCESSING"
        elif "completed" in task_id:
            task["status"] = "COMPLETED"
            if "output" not in task:
                task["output"] = {
                    "url": "https://example.com/mock-video.mp4",
                    "type": "video/mp4"
                }
        elif "failed" in task_id:
            task["status"] = "FAILED"
            if "error" not in task:
                task["error"] = "Task failed due to invalid parameters"
                
        # Return the task response
        return mock_responses.get_runway_task_status_response(
            task_id=task_id,
            status=task["status"],
            include_video="output" in task
        )

class MockRequestsSession:
    """Mock requests session for testing."""
    
    def __init__(self):
        """Initialize the mock requests session."""
        self.base_url = "https://mock-tensorart-api.example.com"
        self.jobs = {}
        self._setup_mock_responses()
        
    def _setup_mock_responses(self):
        """Set up mock responses for testing."""
        self.mock_responses = {
            "POST": {
                f"{self.base_url}/v1/jobs": self._handle_tensorart_job_submission
            },
            "GET": {
                f"{self.base_url}/v1/jobs/": self._handle_tensorart_job_status
            }
        }
        
    def _handle_tensorart_job_submission(self, kwargs):
        """
        Handle TensorArt job submission.
        
        Args:
            kwargs: Request parameters
            
        Returns:
            Response: Mock response
        """
        # Create a mock response
        response = Response()
        response.status_code = 200
        
        # Generate a job ID
        job_id = f"mock-job-{uuid.uuid4().hex[:8]}"
        
        # Create a mock job
        self.jobs[job_id] = {
            "id": job_id,
            "status": "CREATED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z"
        }
        
        # Set the response content
        response._content = json.dumps({
            "jobId": job_id,
            "status": "CREATED"
        }).encode()
        
        return response
        
    def _handle_tensorart_job_status(self, kwargs):
        """
        Handle TensorArt job status request.
        
        Args:
            kwargs: Request parameters
            
        Returns:
            Response: Mock response
        """
        # Create a mock response
        response = Response()
        
        # Get the job ID from the URL
        url = kwargs.get("url", "")
        job_id = url.split("/")[-1]
        
        # Check if the job exists
        if job_id not in self.jobs:
            response.status_code = 404
            response._content = json.dumps({
                "error": f"Job {job_id} not found"
            }).encode()
            return response
        
        # Get the job
        job = self.jobs[job_id]
        
        # Update the job status
        if "pending" in job_id:
            job["status"] = "PENDING"
        elif "processing" in job_id:
            job["status"] = "PROCESSING"
        elif "succeeded" in job_id:
            job["status"] = "SUCCEEDED"
            if "success_info" not in job:
                job["success_info"] = {
                    "images": [
                        {
                            "url": "/Users/dev/womanareoundtheworld/tests/mockdata/generated_clips/mock-image.png",
                            "width": 768,
                            "height": 1280
                        }
                    ]
                }
        elif "failed" in job_id:
            job["status"] = "FAILED"
            if "error" not in job:
                job["error"] = "Job failed due to invalid parameters"
        
        # Set the response content
        response.status_code = 200
        response._content = json.dumps({
            "job": {
                "id": job_id,
                "status": job["status"],
                "createdAt": "2025-04-09T16:29:05.271758",
                "successInfo": {
                    "images": [
                        {
                            "url": "/Users/dev/womanareoundtheworld/tests/mockdata/generated_clips/mock-image.png",
                            "width": 768,
                            "height": 1280
                        }
                    ]
                }
            }
        }).encode()
        
        return response
        
    def post(self, url: str, **kwargs) -> Response:
        """
        Send a POST request.
        
        Args:
            url: Request URL
            **kwargs: Request parameters
            
        Returns:
            Response: Mock response
        """
        # Check if we have a mock handler for this URL
        if url in self.mock_responses["POST"]:
            return self.mock_responses["POST"][url](kwargs)
            
        # Return a 404 response
        response = Response()
        response.status_code = 404
        response._content = json.dumps({
            "error": "Not found"
        }).encode()
        
        return response
        
    def get(self, url: str, **kwargs) -> Response:
        """
        Send a GET request.
        
        Args:
            url: Request URL
            **kwargs: Request parameters
            
        Returns:
            Response: Mock response
        """
        # Check if we have a mock handler for this URL
        for base_url, handler in self.mock_responses["GET"].items():
            if url.startswith(base_url):
                kwargs["url"] = url
                return handler(kwargs)
            
        # Return a 404 response
        response = Response()
        response.status_code = 404
        response._content = json.dumps({
            "error": "Not found"
        }).encode()
        
        return response

# Create a global instance for easy access
mock_session = MockRequestsSession() 