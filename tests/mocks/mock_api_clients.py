"""
Mock API clients for testing without using actual API credits.

This module provides mock clients for various APIs used in the application,
allowing for testing and development without consuming actual API credits.
"""

import uuid
from typing import Any, Dict, Optional

from watw.utils.common.exceptions import (
    RateLimitExceeded,
    RunwayMLError,
    TensorArtError,
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
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self._setup_mock_jobs()

    def _setup_mock_jobs(self) -> None:
        """Set up mock jobs for testing."""
        # Create a few mock jobs with different statuses
        self.jobs["mock-job-pending"] = {
            "id": "mock-job-pending",
            "status": "PENDING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        self.jobs["mock-job-processing"] = {
            "id": "mock-job-processing",
            "status": "PROCESSING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
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
                        "height": 1280,
                    }
                ]
            },
        }

        self.jobs["mock-job-failed"] = {
            "id": "mock-job-failed",
            "status": "FAILED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "error": "Job failed due to invalid parameters",
        }

        self.jobs["mock-job-rate-limited"] = {
            "id": "mock-job-rate-limited",
            "status": "RATE_LIMITED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "retry_after": 60,
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
            raise RateLimitExceeded()

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
            "updated_at": "2023-01-01T00:00:00Z",
        }

        # Store the job
        self.jobs[job_id] = job

        return {
            "job_id": job_id,
            "status": "pending",
            "type": payload.get("type", "unknown"),
            "params": payload,
        }

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
            raise RateLimitExceeded(retry_after=job.get("retry_after", 60))

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
                            "height": 1280,
                        }
                    ]
                }
        elif "failed" in job_id:
            job["status"] = "FAILED"
            if "error" not in job:
                job["error"] = "Job failed due to invalid parameters"

        return {
            "job_id": job_id,
            "status": job["status"].lower(),
            "result": job.get("success_info", {}).get(
                "images", [{"url": "https://mock-tensorart.com/result.mp4"}]
            ),
        }


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
            raise RunwayMLError(
                "The API key you provided is correctly formatted, but it doesn't represent an active key. Has it been deactivated?"
            )

        self.api_key = api_key
        self.base_url = "https://mock-runway-api.example.com"
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._setup_mock_tasks()

    def _setup_mock_tasks(self) -> None:
        """Set up mock tasks for testing."""
        # Create a few mock tasks with different statuses
        self._tasks["mock-task-pending"] = {
            "id": "mock-task-pending",
            "status": "PENDING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        self._tasks["mock-task-processing"] = {
            "id": "mock-task-processing",
            "status": "PROCESSING",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        self._tasks["mock-task-completed"] = {
            "id": "mock-task-completed",
            "status": "COMPLETED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "output": {
                "url": "https://example.com/mock-video.mp4",
                "type": "video/mp4",
            },
        }

        self._tasks["mock-task-failed"] = {
            "id": "mock-task-failed",
            "status": "FAILED",
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
            "error": "Task failed due to invalid parameters",
        }

    def authenticate(self) -> bool:
        """
        Mock authentication method.

        Returns:
            bool: Always returns True for testing.
        """
        return True

    def create_task(self, task_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a mock task.

        Args:
            task_type: Type of task to create
            params: Task parameters

        Returns:
            Dict[str, Any]: Mock task response
        """
        task_id = f"mock-task-{uuid.uuid4().hex[:8]}"

        task = {
            "id": task_id,
            "status": "CREATED",
            "type": task_type,
            "params": params,
            "created_at": "2023-01-01T00:00:00Z",
            "updated_at": "2023-01-01T00:00:00Z",
        }

        self._tasks[task_id] = task

        return {
            "task_id": task_id,
            "status": "pending",
            "type": task_type,
            "params": params,
        }

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get mock task status.

        Args:
            task_id: ID of the task to check

        Returns:
            Dict[str, Any]: Mock task status response
        """
        if task_id not in self._tasks:
            raise RunwayMLError(f"Task {task_id} not found", status_code=404)

        task = self._tasks[task_id]

        # Update status based on task ID
        if "pending" in task_id:
            task["status"] = "PENDING"
        elif "processing" in task_id:
            task["status"] = "PROCESSING"
        elif "completed" in task_id:
            task["status"] = "COMPLETED"
            if "output" not in task:
                task["output"] = {
                    "url": "https://example.com/mock-video.mp4",
                    "type": "video/mp4",
                }
        elif "failed" in task_id:
            task["status"] = "FAILED"
            if "error" not in task:
                task["error"] = "Task failed due to invalid parameters"

        return {
            "task_id": task_id,
            "status": task["status"].lower(),
            "result": task.get(
                "output", {"url": "https://mock-runwayml.com/result.mp4"}
            ),
        }
