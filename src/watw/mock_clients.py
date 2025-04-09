"""
Mock clients for external services.
"""

class MockRunwayMLClient:
    """Mock client for RunwayML API."""
    
    def __init__(self, api_key=None):
        """Initialize the mock RunwayML client.
        
        Args:
            api_key (str, optional): API key for authentication.
        """
        self.api_key = api_key
        
    def authenticate(self):
        """Mock authentication method.
        
        Returns:
            bool: Always returns True for testing.
        """
        return True
        
    def create_task(self, task_type, params):
        """Create a mock task.
        
        Args:
            task_type (str): Type of task to create.
            params (dict): Task parameters.
            
        Returns:
            dict: Mock task response.
        """
        return {
            "task_id": "mock_task_123",
            "status": "pending",
            "type": task_type,
            "params": params
        }
        
    def get_task_status(self, task_id):
        """Get mock task status.
        
        Args:
            task_id (str): ID of the task to check.
            
        Returns:
            dict: Mock task status response.
        """
        return {
            "task_id": task_id,
            "status": "completed",
            "result": {
                "url": "https://mock-runwayml.com/result.mp4"
            }
        }

class MockTensorArtClient:
    """Mock client for TensorArt API."""
    
    def __init__(self, api_key=None):
        """Initialize the mock TensorArt client.
        
        Args:
            api_key (str, optional): API key for authentication.
        """
        self.api_key = api_key
        
    def authenticate(self):
        """Mock authentication method.
        
        Returns:
            bool: Always returns True for testing.
        """
        return True
        
    def create_job(self, job_type, params):
        """Create a mock job.
        
        Args:
            job_type (str): Type of job to create.
            params (dict): Job parameters.
            
        Returns:
            dict: Mock job response.
        """
        return {
            "job_id": "mock_job_456",
            "status": "pending",
            "type": job_type,
            "params": params
        }
        
    def get_job_status(self, job_id):
        """Get mock job status.
        
        Args:
            job_id (str): ID of the job to check.
            
        Returns:
            dict: Mock job status response.
        """
        return {
            "job_id": job_id,
            "status": "completed",
            "result": {
                "url": "https://mock-tensorart.com/result.mp4"
            }
        } 