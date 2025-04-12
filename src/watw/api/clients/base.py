"""
Base API client for Women Around The World project.

This module provides a base class for API clients with common functionality
such as rate limiting, retries, and error handling.
"""

import logging
from typing import Any, Dict, Optional, Union

import requests

from watw.utils.api.rate_limiter import configure_rate_limiter, rate_limited
from watw.utils.api.retry import RetryConfig, with_retry
from watw.api.config import Config
from watw.utils.common.exceptions import APIError, ConfigurationError


class APITimeoutError(APIError):
    """Exception raised when an API request times out."""

    def __init__(
        self,
        message: Optional[str] = None,
        service: Optional[str] = None,
        timeout: Optional[Union[int, float]] = None,
        response: Optional[requests.Response] = None
    ):
        if message is None:
            message = "Request timed out"
            if service:
                message += f" for service {service}"
            if timeout is not None:
                message += f" after {timeout} seconds"
        super().__init__(message, response)
        self.service = service
        self.timeout = timeout


class BaseAPIClient:
    """Base class for API clients."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        service_name: str = "base",
        require_api_key: bool = True,
    ):
        """
        Initialize the base API client.

        Args:
            api_key: API key for authentication
            base_url: Base URL for API requests
            service_name: Name of the service
            require_api_key: Whether to require a valid API key
        """
        self.api_key = api_key
        self.base_url = base_url
        self.service_name = service_name
        self.require_api_key = require_api_key
        self.logger = logging.getLogger(__name__)

        # Validate configuration
        if require_api_key and not api_key:
            raise ConfigurationError(f"{service_name} API key is required")
        if not base_url:
            raise ConfigurationError(f"{service_name} base URL is required")

    def get_headers(self) -> Dict[str, str]:
        """Get default headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def get(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Any:
        """
        Make a GET request.

        Args:
            endpoint: API endpoint
            params: Query parameters
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        headers = headers or {}
        headers.update(self.get_headers())

        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"GET request failed: {str(e)}")
            raise APIError(f"GET request failed: {str(e)}")

    def post(
        self,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 30,
    ) -> Any:
        """
        Make a POST request.

        Args:
            endpoint: API endpoint
            json: JSON data
            data: Form data
            headers: Request headers
            timeout: Request timeout in seconds

        Returns:
            Response data
        """
        url = f"{self.base_url}{endpoint}"
        headers = headers or {}
        headers.update(self.get_headers())

        try:
            response = requests.post(
                url,
                json=json,
                data=data,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"POST request failed: {str(e)}")
            raise APIError(f"POST request failed: {str(e)}")

    def put(
        self, endpoint: str, json: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> requests.Response:
        """
        Make a PUT request.

        Args:
            endpoint: API endpoint path
            json: JSON body for the request
            **kwargs: Additional arguments to pass to request()

        Returns:
            requests.Response: API response
        """
        return self.request("PUT", endpoint, json=json, **kwargs)

    def delete(self, endpoint: str, **kwargs: Any) -> requests.Response:
        """
        Make a DELETE request.

        Args:
            endpoint: API endpoint path
            **kwargs: Additional arguments to pass to request()

        Returns:
            requests.Response: API response
        """
        return self.request("DELETE", endpoint, **kwargs)

    @rate_limited(service="base")
    @with_retry()
    def request(
        self,
        method: str,
        endpoint: str,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        retry_config: Optional[RetryConfig] = None,
    ) -> requests.Response:
        """
        Make an API request with retries and rate limiting.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json: JSON body for the request
            params: Query parameters
            headers: Additional headers
            files: Files to upload
            timeout: Request timeout in seconds
            retry_config: Custom retry configuration

        Returns:
            requests.Response: API response

        Raises:
            requests.RequestException: If the request fails
        """
        # Prepare request
        url = f"{self.base_url}{endpoint}"
        request_headers = self.get_headers()

        # Add custom headers
        if headers:
            request_headers.update(headers)

        # Use custom timeout if provided
        request_timeout = timeout or 30

        # Log request
        self.logger.debug(f"Making {method} request to {url}")

        # Make request
        response = requests.request(
            method=method,
            url=url,
            json=json,
            params=params,
            headers=request_headers,
            files=files,
            timeout=request_timeout,
        )

        # Check response
        if not response.ok:
            self.logger.warning(
                f"Request failed: {response.status_code} - {response.text}"
            )
            response.raise_for_status()

        return response
