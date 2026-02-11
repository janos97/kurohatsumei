"""Meshy API client for image-to-3D conversion."""

import base64
import os
import time
from pathlib import Path
from typing import Any

import requests

from ..config import get_config


class MeshyClient:
    """Client for Meshy.ai API."""

    def __init__(self, api_key: str | None = None):
        config = get_config()
        self.api_key = api_key or config.meshy_api_key or os.environ.get("MESHY_API_KEY", "")
        self.api_url = config.meshy_api_url
        self.poll_interval = config.meshy_poll_interval

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def image_to_3d(
        self,
        image_path: str | Path,
        callback: callable | None = None,
    ) -> tuple[bool, bytes | str]:
        """Convert an image to a 3D model.

        Args:
            image_path: Path to the input image.
            callback: Optional callback for progress updates.

        Returns:
            Tuple of (success, result). On success, result is GLB file bytes.
        """
        if not self.api_key:
            return False, "Meshy API key not configured. Set MESHY_API_KEY environment variable."

        try:
            # Read and encode the image
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Determine image type
            ext = Path(image_path).suffix.lower()
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            mime_type = mime_types.get(ext, "image/png")

            # Create data URL
            image_base64 = base64.b64encode(image_data).decode()
            image_url = f"data:{mime_type};base64,{image_base64}"

            # Create the task
            if callback:
                callback("Creating Meshy task...")

            response = requests.post(
                f"{self.api_url}/v2/image-to-3d",
                headers=self._get_headers(),
                json={
                    "image_url": image_url,
                    "enable_pbr": False,  # We don't need PBR for 3D printing
                },
                timeout=60,
            )
            response.raise_for_status()

            task_data = response.json()
            task_id = task_data.get("result")

            if not task_id:
                return False, "Failed to create Meshy task: no task ID returned"

            # Poll for completion
            return self._poll_task(task_id, callback)

        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except requests.exceptions.RequestException as e:
            return False, f"Meshy API error: {str(e)}"
        except Exception as e:
            return False, f"Meshy conversion failed: {str(e)}"

    def _poll_task(
        self,
        task_id: str,
        callback: callable | None = None,
        timeout: float = 600,
    ) -> tuple[bool, bytes | str]:
        """Poll a Meshy task until completion.

        Args:
            task_id: The task ID to poll.
            callback: Optional callback for progress updates.
            timeout: Maximum wait time in seconds.

        Returns:
            Tuple of (success, result). On success, result is GLB file bytes.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.api_url}/v2/image-to-3d/{task_id}",
                    headers=self._get_headers(),
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()
                status = data.get("status", "")
                progress = data.get("progress", 0)

                if callback:
                    callback(f"Meshy: {status} ({progress}%)")

                if status == "SUCCEEDED":
                    # Get the GLB URL
                    model_urls = data.get("model_urls", {})
                    glb_url = model_urls.get("glb")

                    if not glb_url:
                        return False, "No GLB URL in completed task"

                    # Download the GLB
                    if callback:
                        callback("Downloading 3D model...")

                    glb_response = requests.get(glb_url, timeout=60)
                    glb_response.raise_for_status()

                    return True, glb_response.content

                elif status == "FAILED":
                    error = data.get("task_error", {})
                    message = error.get("message", "Unknown error")
                    return False, f"Meshy task failed: {message}"

                elif status == "EXPIRED":
                    return False, "Meshy task expired"

            except requests.exceptions.RequestException as e:
                if callback:
                    callback(f"Polling error (retrying): {str(e)}")

            time.sleep(self.poll_interval)

        return False, f"Meshy task timed out after {timeout} seconds"

    def check_connection(self) -> tuple[bool, str]:
        """Check if Meshy API is accessible.

        Returns:
            Tuple of (success, message).
        """
        if not self.api_key:
            return False, "Meshy API key not configured"

        try:
            response = requests.get(
                f"{self.api_url}/v2/image-to-3d",
                headers=self._get_headers(),
                params={"page_size": 1},
                timeout=10,
            )
            response.raise_for_status()
            return True, "Connected to Meshy API"
        except requests.exceptions.RequestException as e:
            return False, f"Meshy API check failed: {str(e)}"
