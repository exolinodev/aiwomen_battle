"""
API response validation functions.

This module provides functions for validating responses from various APIs.
"""

from typing import Any, Dict

from .exceptions import ValidationError


def validate_tensorart_job_response(response: Dict[str, Any]) -> None:
    """Validate the response from a TensorArt job submission."""
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")

    if "job_id" not in response:
        raise ValidationError("Response must contain a job_id")

    if not isinstance(response["job_id"], str):
        raise ValidationError("job_id must be a string")


def validate_tensorart_job_status_response(response: Dict[str, Any]) -> None:
    """Validate the response from a TensorArt job status check."""
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")

    if "status" not in response:
        raise ValidationError("Response must contain a status")

    if not isinstance(response["status"], str):
        raise ValidationError("status must be a string")

    if "output" not in response:
        raise ValidationError("Response must contain an output")

    if not isinstance(response["output"], dict):
        raise ValidationError("output must be a dictionary")

    if "url" not in response["output"]:
        raise ValidationError("output must contain a url")

    if not isinstance(response["output"]["url"], str):
        raise ValidationError("url must be a string")


def validate_runway_task_response(response: Dict[str, Any]) -> None:
    """Validate the response from a RunwayML task creation."""
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")

    if "task_id" not in response:
        raise ValidationError("Response must contain a task_id")

    if not isinstance(response["task_id"], str):
        raise ValidationError("task_id must be a string")


def validate_runway_task_status_response(response: Dict[str, Any]) -> None:
    """Validate the response from a RunwayML task status check."""
    if not isinstance(response, dict):
        raise ValidationError("Response must be a dictionary")

    if "status" not in response:
        raise ValidationError("Response must contain a status")

    if not isinstance(response["status"], str):
        raise ValidationError("status must be a string")

    if "output" not in response:
        raise ValidationError("Response must contain an output")

    if not isinstance(response["output"], dict):
        raise ValidationError("output must be a dictionary")

    if "url" not in response["output"]:
        raise ValidationError("output must contain a url")

    if not isinstance(response["output"]["url"], str):
        raise ValidationError("url must be a string")
