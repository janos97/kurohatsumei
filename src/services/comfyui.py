"""ComfyUI API client for image generation."""
from __future__ import annotations

import json
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from ..config import get_config


class ComfyUIClient:
    """Client for ComfyUI API."""

    def __init__(self, url: str | None = None, workflow_path: str | None = None):
        config = get_config()
        self.url = url or config.comfyui_url
        self.workflow_path = workflow_path or config.comfyui_workflow_path
        self.poll_interval = config.comfyui_poll_interval
        self.num_images = config.comfyui_num_images
        self.client_id = str(uuid.uuid4())

    def load_workflow(self) -> dict[str, Any]:
        """Load the workflow JSON file.

        Returns:
            Parsed workflow dictionary.
        """
        with open(self.workflow_path) as f:
            return json.load(f)

    def inject_prompt(self, workflow: dict[str, Any], prompt: str) -> dict[str, Any]:
        """Inject prompt into the workflow's CLIPTextEncode node.

        Searches for nodes of type 'CLIPTextEncode' and updates the text input.
        Typically, the positive prompt node is connected to the conditioning input.

        Args:
            workflow: The workflow dictionary.
            prompt: The prompt text to inject.

        Returns:
            Modified workflow.
        """
        modified = False

        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                inputs = node.get("inputs", {})
                # Look for the positive prompt (usually has "positive" in title or is first)
                # Most workflows have the positive prompt node with just "text" input
                if "text" in inputs:
                    node["inputs"]["text"] = prompt
                    modified = True
                    break  # Only modify the first (positive) CLIPTextEncode

        if not modified:
            raise ValueError("Could not find CLIPTextEncode node in workflow")

        return workflow

    def set_seed(self, workflow: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        """Set a random seed in KSampler nodes.

        Args:
            workflow: The workflow dictionary.
            seed: Seed value. If None, uses current timestamp.

        Returns:
            Modified workflow.
        """
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)

        for node_id, node in workflow.items():
            if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                if "inputs" in node and "seed" in node["inputs"]:
                    node["inputs"]["seed"] = seed

        return workflow

    def queue_prompt(self, workflow: dict[str, Any]) -> str:
        """Queue a workflow for execution.

        Args:
            workflow: The workflow dictionary.

        Returns:
            The prompt_id for tracking.
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }

        response = requests.post(
            f"{self.url}/prompt",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        return data["prompt_id"]

    def poll_for_completion(
        self,
        prompt_id: str,
        timeout: float = 300,
        callback: callable | None = None,
    ) -> tuple[bool, dict[str, Any] | str]:
        """Poll for workflow completion.

        Args:
            prompt_id: The prompt ID to track.
            timeout: Maximum wait time in seconds.
            callback: Optional callback(status_msg) for progress updates.

        Returns:
            Tuple of (success, result). On success, result is the history entry.
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{self.url}/history/{prompt_id}",
                    timeout=10,
                )
                response.raise_for_status()

                history = response.json()

                if prompt_id in history:
                    entry = history[prompt_id]
                    status = entry.get("status", {})

                    if status.get("completed", False):
                        return True, entry
                    elif status.get("status_str") == "error":
                        messages = status.get("messages", [])
                        error_msg = messages[-1] if messages else "Unknown error"
                        return False, f"ComfyUI error: {error_msg}"

                if callback:
                    elapsed = int(time.time() - start_time)
                    callback(f"Generating... ({elapsed}s)")

            except requests.exceptions.RequestException as e:
                if callback:
                    callback(f"Polling error (retrying): {str(e)}")

            time.sleep(self.poll_interval)

        return False, f"Timeout after {timeout} seconds"

    def fetch_images(self, history_entry: dict[str, Any]) -> list[Image.Image]:
        """Fetch generated images from ComfyUI.

        Args:
            history_entry: The history entry from poll_for_completion.

        Returns:
            List of PIL Images.
        """
        images = []
        outputs = history_entry.get("outputs", {})

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    folder_type = img_info.get("type", "output")

                    params = {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": folder_type,
                    }

                    response = requests.get(
                        f"{self.url}/view",
                        params=params,
                        timeout=30,
                    )
                    response.raise_for_status()

                    img = Image.open(BytesIO(response.content))
                    images.append(img)

        return images

    def generate_images(
        self,
        prompt: str,
        num_images: int | None = None,
        callback: callable | None = None,
    ) -> tuple[bool, list[Image.Image] | str]:
        """Generate images from a text prompt.

        Args:
            prompt: The text prompt for image generation.
            num_images: Number of images to generate. Defaults to config value.
            callback: Optional callback for progress updates.

        Returns:
            Tuple of (success, result). On success, result is list of PIL Images.
        """
        if num_images is None:
            num_images = self.num_images

        all_images = []

        try:
            workflow = self.load_workflow()

            for i in range(num_images):
                if callback:
                    callback(f"Generating image {i + 1}/{num_images}...")

                # Inject prompt and set unique seed
                modified = self.inject_prompt(workflow.copy(), prompt)
                modified = self.set_seed(modified)

                # Queue and wait
                prompt_id = self.queue_prompt(modified)

                def progress_callback(msg: str) -> None:
                    if callback:
                        callback(f"Image {i + 1}/{num_images}: {msg}")

                success, result = self.poll_for_completion(
                    prompt_id,
                    callback=progress_callback,
                )

                if not success:
                    return False, result

                # Fetch the image
                images = self.fetch_images(result)
                all_images.extend(images)

            return True, all_images

        except FileNotFoundError:
            return False, f"Workflow file not found: {self.workflow_path}"
        except json.JSONDecodeError:
            return False, f"Invalid JSON in workflow file: {self.workflow_path}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to ComfyUI at {self.url}"
        except requests.exceptions.RequestException as e:
            return False, f"ComfyUI request failed: {str(e)}"
        except Exception as e:
            return False, f"Image generation failed: {str(e)}"

    def check_connection(self) -> tuple[bool, str]:
        """Check if ComfyUI is reachable.

        Returns:
            Tuple of (success, message).
        """
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=5)
            response.raise_for_status()
            return True, f"Connected to ComfyUI at {self.url}"
        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to ComfyUI at {self.url}"
        except requests.exceptions.RequestException as e:
            return False, f"ComfyUI check failed: {str(e)}"
