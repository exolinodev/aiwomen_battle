"""
Test utilities for using mock API responses.

This module provides utilities for testing with mock API responses,
allowing for testing and development without consuming actual API credits.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Optional, Union, cast
from unittest.mock import MagicMock, patch

from tests.mocks.mock_api_clients import MockRunwayMLClient
from tests.mocks.mock_api_responses import MockResponses
from tests.test_config import TEST_CONFIG


class MockAPITestCase(unittest.TestCase):
    """
    Base test case for tests using mock API responses.

    This class provides utilities for testing with mock API responses,
    allowing for testing and development without consuming actual API credits.
    """

    def setUp(self) -> None:
        """Set up the test case."""
        super().setUp()

        # Use configured mock data directories
        self.mock_data_dir = Path(str(TEST_CONFIG["mock_data_dir"]))
        self.mock_voiceovers_dir = Path(str(TEST_CONFIG["mock_voiceovers_dir"]))
        self.mock_generated_clips_dir = Path(str(TEST_CONFIG["mock_generated_clips_dir"]))
        self.mock_temp_files_dir = Path(str(TEST_CONFIG["mock_temp_files_dir"]))
        self.mock_final_video_dir = Path(str(TEST_CONFIG["mock_final_video_dir"]))

        # Create a temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp(dir=self.mock_temp_files_dir)
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(exist_ok=True)

        # Create mock image and video files
        self.mock_image_path = MockResponses.create_mock_image(
            width=768, height=1280, output_path=self.output_dir / "mock_image.png"
        )

        self.mock_video_path = MockResponses.create_mock_video(
            duration=5, output_path=self.output_dir / "mock_video.mp4"
        )

        # Create mock session
        mock_session = MagicMock()
        mock_session.get.return_value = MagicMock()
        mock_session.post.return_value = MagicMock()

        # Patch requests.Session
        self.requests_patcher = patch("requests.Session", return_value=mock_session)
        self.requests_patcher.start()

        # Patch RunwayML client
        self.runwayml_patcher = patch(
            "runwayml.RunwayML", return_value=MockRunwayMLClient()
        )
        self.runwayml_patcher.start()

    def tearDown(self) -> None:
        """Tear down the test case."""
        # Stop patchers
        self.requests_patcher.stop()
        self.runwayml_patcher.stop()

        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

        super().tearDown()

    def assert_file_exists(self, file_path: Union[str, Path]) -> None:
        """
        Assert that a file exists.

        Args:
            file_path: Path to the file
        """
        self.assertTrue(os.path.exists(file_path), f"File {file_path} does not exist")

    def assert_file_not_exists(self, file_path: Union[str, Path]) -> None:
        """
        Assert that a file does not exist.

        Args:
            file_path: Path to the file
        """
        self.assertFalse(os.path.exists(file_path), f"File {file_path} exists")

    def assert_directory_exists(self, directory_path: Union[str, Path]) -> None:
        """
        Assert that a directory exists.

        Args:
            directory_path: Path to the directory
        """
        self.assertTrue(
            os.path.isdir(directory_path), f"Directory {directory_path} does not exist"
        )

    def assert_directory_not_exists(self, directory_path: Union[str, Path]) -> None:
        """
        Assert that a directory does not exist.

        Args:
            directory_path: Path to the directory
        """
        self.assertFalse(
            os.path.isdir(directory_path), f"Directory {directory_path} exists"
        )

    def assert_file_size(self, file_path: Union[str, Path], min_size: int = 0) -> None:
        """
        Assert that a file has a minimum size.

        Args:
            file_path: Path to the file
            min_size: Minimum file size in bytes
        """
        self.assert_file_exists(file_path)
        self.assertGreaterEqual(
            os.path.getsize(file_path),
            min_size,
            f"File {file_path} is smaller than {min_size} bytes",
        )

    def assert_response_status_code(self, response: Any, status_code: int) -> None:
        """
        Assert that a response has a specific status code.

        Args:
            response: Response object
            status_code: Expected status code
        """
        self.assertEqual(
            response.status_code,
            status_code,
            f"Response status code is {response.status_code}, expected {status_code}",
        )

    def assert_response_json(self, response: Any, expected_json: Dict[str, Any]) -> None:
        """
        Assert that a response has specific JSON content.

        Args:
            response: Response object
            expected_json: Expected JSON content
        """
        self.assertEqual(
            response.json(),
            expected_json,
            f"Response JSON is {response.json()}, expected {expected_json}",
        )

    def assert_response_contains(
        self, response: Any, key: str, value: Any = None
    ) -> None:
        """
        Assert that a response contains a specific key and optionally a specific value.

        Args:
            response: Response object
            key: Key to check for
            value: Expected value (optional)
        """
        response_json = response.json()
        self.assertIn(key, response_json, f"Response JSON does not contain key {key}")

        if value is not None:
            self.assertEqual(
                response_json[key],
                value,
                f"Response JSON key {key} is {response_json[key]}, expected {value}",
            )

    def assert_response_contains_list(
        self, response: Any, key: str, min_length: int = 1
    ) -> None:
        """
        Assert that a response contains a list with a minimum length.

        Args:
            response: Response object
            key: Key to check for
            min_length: Minimum list length
        """
        response_json = response.json()
        self.assertIn(key, response_json, f"Response JSON does not contain key {key}")
        self.assertIsInstance(
            response_json[key], list, f"Response JSON key {key} is not a list"
        )
        self.assertGreaterEqual(
            len(response_json[key]),
            min_length,
            f"Response JSON key {key} list length is {len(response_json[key])}, expected at least {min_length}",
        )


def mock_api_response(
    response_data: Dict[str, Any], status_code: int = 200
) -> MagicMock:
    """
    Create a mock API response.

    Args:
        response_data: Response data
        status_code: Response status code

    Returns:
        Mock response object
    """
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_data
    return mock_response


def mock_api_error(status_code: int, error_message: str) -> MagicMock:
    """
    Create a mock API error response.

    Args:
        status_code: Error status code
        error_message: Error message

    Returns:
        Mock error response object
    """
    return mock_api_response({"error": error_message}, status_code)


def mock_rate_limit_error(retry_after: int = 60) -> MagicMock:
    """
    Create a mock rate limit error response.

    Args:
        retry_after: Seconds to wait before retrying

    Returns:
        Mock rate limit error response object
    """
    return mock_api_response(
        {"error": f"Rate limit exceeded. Retry after {retry_after} seconds"},
        429,
    )


def mock_tensorart_job_response(
    job_id: str = "test_job", status: str = "CREATED"
) -> MagicMock:
    """
    Create a mock TensorArt job response.

    Args:
        job_id: Job ID
        status: Job status

    Returns:
        Mock job response object
    """
    return mock_api_response(MockResponses.get_tensorart_job_response(job_id))


def mock_tensorart_job_status_response(
    job_id: str = "test_job",
    status: str = "SUCCEEDED",
    include_image: bool = True,
) -> MagicMock:
    """
    Create a mock TensorArt job status response.

    Args:
        job_id: Job ID
        status: Job status
        include_image: Whether to include image in output

    Returns:
        Mock job status response object
    """
    return mock_api_response(
        MockResponses.get_tensorart_job_status_response(job_id, status, include_image)
    )


def mock_runway_task_response(
    task_id: str = "test_task", status: str = "CREATED"
) -> MagicMock:
    """
    Create a mock RunwayML task response.

    Args:
        task_id: Task ID
        status: Task status

    Returns:
        Mock task response object
    """
    return mock_api_response(MockResponses.get_runway_task_response(task_id))


def mock_runway_task_status_response(
    task_id: str = "test_task",
    status: str = "COMPLETED",
    include_video: bool = True,
) -> MagicMock:
    """
    Create a mock RunwayML task status response.

    Args:
        task_id: Task ID
        status: Task status
        include_video: Whether to include video in output

    Returns:
        Mock task status response object
    """
    return mock_api_response(
        MockResponses.get_runway_task_status_response(task_id, status, include_video)
    )


def create_mock_image(
    width: int = 768,
    height: int = 1280,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a mock image for testing.

    Args:
        width: Image width
        height: Image height
        output_path: Output path for the image

    Returns:
        Path to the created image
    """
    return MockResponses.create_mock_image(width, height, output_path)


def create_mock_video(
    duration: int = 5,
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Create a mock video for testing.

    Args:
        duration: Video duration in seconds
        output_path: Output path for the video

    Returns:
        Path to the created video
    """
    return MockResponses.create_mock_video(duration=duration, output_path=output_path)
