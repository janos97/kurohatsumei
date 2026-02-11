"""Stage 1: Prompt engineering using Ollama."""

from pathlib import Path
from typing import Any

from ..services.ollama import OllamaClient
from ..utils.file_manager import save_json


def generate_prompts(
    description: str,
    project_folder: Path | str | None = None,
) -> tuple[bool, dict[str, str] | str]:
    """Generate image and 3D prompts from user description.

    Args:
        description: User's description of what they want to print.
        project_folder: Optional folder to save prompts.json.

    Returns:
        Tuple of (success, result). On success, result is dict with
        'image_prompt' and 'threed_hint'. On failure, result is error message.
    """
    client = OllamaClient()

    success, result = client.generate_prompts(description)

    if success and project_folder:
        # Save the prompts to the project folder
        save_json(project_folder, "prompts.json", {
            "user_description": description,
            "image_prompt": result["image_prompt"],
            "threed_hint": result["threed_hint"],
        })

    return success, result


def check_ollama() -> tuple[bool, str]:
    """Check if Ollama is available.

    Returns:
        Tuple of (success, message).
    """
    client = OllamaClient()
    return client.check_connection()
