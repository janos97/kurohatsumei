"""Ollama API client for prompt engineering."""

import json
from typing import Any

import requests

from ..config import get_config


SYSTEM_PROMPT = """You are an expert at creating prompts for text-to-image AI models,
specifically optimized for generating images that will be converted to 3D printable models.

Given a user's description of what they want to 3D print, you must:

1. Create an "image_prompt" that will generate a clear, well-defined image suitable for
   3D reconstruction. The prompt should:
   - Describe the object from a 3/4 angle view (slightly above, showing front and one side)
   - Use neutral, solid background (white, gray, or gradient)
   - Emphasize clean edges and clear silhouettes
   - Avoid complex textures that won't translate to 3D printing
   - Include "studio lighting" or "soft shadows" for depth perception
   - Avoid fine details smaller than 0.5mm that won't print well on resin printers
   - Describe the object as solid and self-supporting (no floating parts)

2. Create a "threed_hint" that provides guidance for the 3D reconstruction stage:
   - Describe the expected geometry (organic, geometric, symmetrical, etc.)
   - Note any areas that might need support structures
   - Suggest orientation for optimal printing

You must respond with valid JSON containing exactly these two fields:
- "image_prompt": A detailed prompt for the image generation model
- "threed_hint": Guidance for 3D reconstruction

Be concise but specific. Focus on printability."""


class OllamaClient:
    """Client for Ollama API with structured JSON output."""

    def __init__(self, url: str | None = None, model: str | None = None):
        config = get_config()
        self.url = url or config.ollama_url
        self.model = model or config.ollama_model

    def generate_prompts(self, description: str) -> tuple[bool, dict[str, str] | str]:
        """Generate image and 3D prompts from user description.

        Args:
            description: User's description of what they want to print.

        Returns:
            Tuple of (success, result). On success, result is dict with
            'image_prompt' and 'threed_hint'. On failure, result is error message.
        """
        schema = {
            "type": "object",
            "properties": {
                "image_prompt": {"type": "string"},
                "threed_hint": {"type": "string"},
            },
            "required": ["image_prompt", "threed_hint"],
        }

        try:
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": f"Create prompts for 3D printing this object: {description}",
                    "system": SYSTEM_PROMPT,
                    "format": schema,
                    "stream": False,
                },
                timeout=120,
            )
            response.raise_for_status()

            data = response.json()
            result_text = data.get("response", "")

            # Parse the JSON response
            result = json.loads(result_text)

            if "image_prompt" not in result or "threed_hint" not in result:
                return False, "Invalid response: missing required fields"

            return True, result

        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama at {self.url}. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Ollama request timed out"
        except requests.exceptions.RequestException as e:
            return False, f"Ollama request failed: {str(e)}"
        except json.JSONDecodeError as e:
            return False, f"Failed to parse Ollama response as JSON: {str(e)}"

    def check_connection(self) -> tuple[bool, str]:
        """Check if Ollama is reachable and the model is available.

        Returns:
            Tuple of (success, message).
        """
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            response.raise_for_status()

            # Check if the model is available
            data = response.json()
            models = [m.get("name", "") for m in data.get("models", [])]

            # Handle model names with and without tags
            model_base = self.model.split(":")[0]
            available = any(
                m == self.model or m.startswith(f"{model_base}:")
                for m in models
            )

            if available:
                return True, f"Connected to Ollama, model '{self.model}' available"
            else:
                return False, f"Model '{self.model}' not found. Available: {', '.join(models)}"

        except requests.exceptions.ConnectionError:
            return False, f"Cannot connect to Ollama at {self.url}"
        except requests.exceptions.RequestException as e:
            return False, f"Ollama check failed: {str(e)}"
