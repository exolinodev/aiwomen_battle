"""
Test file for mock API responses.

This file demonstrates how to use the mock API responses for testing.
"""

import unittest
from pathlib import Path

import requests

from tests.mocks.mock_api_clients import MockRunwayMLClient, MockTensorArtClient
from tests.mocks.mock_api_responses import mock_responses
from watw.utils.common.test_utils import MockAPITestCase, mock_api_error


class TestMockAPIResponses(MockAPITestCase):
    """Test class for mock API responses."""

    def test_mock_tensorart_job_response(self):
        """Test mock TensorArt job response."""
        # Create a mock job response
        job_id = "mock-job-12345"
        response = mock_responses.get_tensorart_job_response(job_id)

        # Assert response properties
        self.assertEqual(response["jobId"], job_id)
        self.assertEqual(response["status"], "CREATED")

    def test_mock_tensorart_job_status_response(self):
        """Test mock TensorArt job status response."""
        # Create a mock job status response
        job_id = "mock-job-12345"
        response = mock_responses.get_tensorart_job_status_response(job_id)

        # Assert response properties
        self.assertEqual(response["job"]["id"], job_id)
        self.assertEqual(response["job"]["status"], "SUCCEEDED")
        self.assertIn("successInfo", response["job"])
        self.assertIn("images", response["job"]["successInfo"])
        self.assertGreaterEqual(len(response["job"]["successInfo"]["images"]), 1)

    def test_mock_runway_task_response(self):
        """Test mock RunwayML task response."""
        # Create a mock task response
        task_id = "mock-task-67890"
        response = mock_responses.get_runway_task_response(task_id)

        # Assert response properties
        self.assertEqual(response["id"], task_id)
        self.assertEqual(response["status"], "CREATED")

    def test_mock_runway_task_status_response(self):
        """Test mock RunwayML task status response."""
        # Create a mock task status response
        task_id = "mock-task-67890"
        response = mock_responses.get_runway_task_status_response(
            task_id, "COMPLETED", True
        )

        # Assert response properties
        self.assertEqual(response["id"], task_id)
        self.assertEqual(response["status"], "COMPLETED")
        self.assertIn("output", response)
        self.assertGreaterEqual(len(response["output"]), 1)

    def test_mock_rate_limit_error(self):
        """Test mock rate limit error response."""
        # Create a mock rate limit error response
        retry_after = 60
        response = mock_responses.get_rate_limit_response(retry_after)

        # Assert response properties
        self.assertIn("error", response)
        self.assertIn("retry_after", response)
        self.assertEqual(response["retry_after"], retry_after)

    def test_mock_api_error(self):
        """Test mock API error response."""
        # Create a mock API error response
        status_code = 400
        error_message = "Invalid parameters provided"
        response = mock_api_error(status_code, error_message)

        # Assert response properties
        self.assertEqual(response.status_code, status_code)
        self.assertEqual(response.json()["error"], error_message)


class TestMockAPIClients(MockAPITestCase):
    """Test class for mock API clients."""

    def test_mock_tensorart_client(self):
        """Test mock TensorArt client."""
        # Create a mock TensorArt client
        client = MockTensorArtClient()

        # Submit a job
        payload = {
            "request_id": "test-request-12345",
            "stages": [
                {
                    "type": "INPUT_INITIALIZE",
                    "inputInitialize": {"seed": -1, "count": 1},
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": 768,
                        "height": 1280,
                        "prompts": [
                            {"text": "A beautiful landscape", "weight": 1.0},
                            {"text": "Blurry, low quality", "weight": -1.0},
                        ],
                        "sampler": "Euler",
                        "sdVae": "Automatic",
                        "steps": 30,
                        "sd_model": "757279507095956705",
                        "clip_skip": 2,
                        "cfg_scale": 7,
                    },
                },
            ],
        }

        response = client.submit_job(payload)

        # Assert response properties
        self.assertIn("jobId", response)
        self.assertIn("status", response)
        self.assertEqual(response["status"], "CREATED")

        # Get job status
        job_id = response["jobId"]
        status_response = client.get_job_status(job_id)

        # Assert status response properties
        self.assertIn("job", status_response)
        self.assertIn("id", status_response["job"])
        self.assertEqual(status_response["job"]["id"], job_id)

    def test_mock_runwayml_client(self):
        """Test mock RunwayML client."""
        # Create a mock RunwayML client
        client = MockRunwayMLClient()

        # Create a task
        task_response = client.image_to_video().create(
            model="gen3a_turbo",
            prompt_image="data:image/png;base64,mock-base64-image",
            prompt_text="Create a smooth, cinematic animation",
            duration=5,
            ratio="768:1280",
            seed=12345,
            watermark=False,
        )

        # Assert response properties
        self.assertIn("id", task_response)
        self.assertIn("status", task_response)
        self.assertEqual(task_response["status"], "CREATED")

        # Get task status
        task_id = task_response["id"]
        status_response = client.tasks.retrieve(task_id)

        # Assert status response properties
        self.assertIn("id", status_response)
        self.assertEqual(status_response["id"], task_id)

    def test_mock_requests_session(self):
        """Test mock requests session."""
        # Create a mock requests session
        session = requests.Session()

        # Send a POST request to TensorArt API
        response = session.post(
            "https://mock-tensorart-api.example.com/v1/jobs",
            json={
                "request_id": "test-request-12345",
                "stages": [
                    {
                        "type": "INPUT_INITIALIZE",
                        "inputInitialize": {"seed": -1, "count": 1},
                    },
                    {
                        "type": "DIFFUSION",
                        "diffusion": {
                            "width": 768,
                            "height": 1280,
                            "prompts": [
                                {"text": "A beautiful landscape", "weight": 1.0},
                                {"text": "Blurry, low quality", "weight": -1.0},
                            ],
                            "sampler": "Euler",
                            "sdVae": "Automatic",
                            "steps": 30,
                            "sd_model": "757279507095956705",
                            "clip_skip": 2,
                            "cfg_scale": 7,
                        },
                    },
                ],
            },
        )

        # Assert response properties
        self.assertEqual(response.status_code, 200)
        self.assertIn("jobId", response.json())
        self.assertEqual(response.json()["status"], "CREATED")

        # Send a GET request to TensorArt API
        job_id = response.json()["jobId"]
        response = session.get(
            f"https://mock-tensorart-api.example.com/v1/jobs/{job_id}"
        )

        # Assert response properties
        self.assertEqual(response.status_code, 200)
        self.assertIn("job", response.json())
        self.assertEqual(response.json()["job"]["id"], job_id)

    def test_mock_runwayml_requests(self):
        """Test mock RunwayML requests."""
        # Create a mock RunwayML client
        client = MockRunwayMLClient(api_key="key_mock_api_key")

        # Create a task
        task_response = client.image_to_video().create(
            model="gen3a_turbo",
            prompt_image="data:image/png;base64,mock-base64-image",
            prompt_text="Create a smooth, cinematic animation",
            duration=5,
            ratio="768:1280",
            seed=12345,
            watermark=False,
        )

        # Assert response properties
        self.assertIn("id", task_response)
        self.assertIn("status", task_response)
        self.assertEqual(task_response["status"], "CREATED")

        # Get task status
        task_id = task_response["id"]
        status_response = client.tasks.retrieve(task_id)

        # Assert status response properties
        self.assertIn("id", status_response)
        self.assertEqual(status_response["id"], task_id)


class TestMockFileOperations(MockAPITestCase):
    """Test class for mock file operations."""

    def test_create_mock_image(self):
        """Test creating a mock image file."""
        # Create a temporary directory
        temp_dir = self.output_dir

        # Create a mock image file
        image_path = Path(temp_dir) / "mock-image.png"
        mock_responses.create_mock_image(width=768, height=1280, output_path=image_path)

        # Assert file exists and has content
        self.assertTrue(image_path.exists())
        self.assertGreater(image_path.stat().st_size, 0)

    def test_create_mock_video(self):
        """Test creating a mock video file."""
        # Create a temporary directory
        temp_dir = self.output_dir

        # Create a mock video file
        video_path = Path(temp_dir) / "mock-video.mp4"
        mock_responses.create_mock_video(duration=5, output_path=video_path)

        # Assert file exists and has content
        self.assertTrue(video_path.exists())
        self.assertGreater(video_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
