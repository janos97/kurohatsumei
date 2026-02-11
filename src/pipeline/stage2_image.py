"""Stage 2: Image generation using ComfyUI."""

from pathlib import Path
from typing import Any

from PIL import Image

from ..services.comfyui import ComfyUIClient
from ..utils.file_manager import get_project_path


def generate_images(
    prompt: str,
    project_folder: Path | str | None = None,
    num_images: int | None = None,
    callback: callable | None = None,
) -> tuple[bool, list[Image.Image] | str]:
    """Generate images from a text prompt.

    Args:
        prompt: The text prompt for image generation.
        project_folder: Optional folder to save generated images.
        num_images: Number of images to generate.
        callback: Optional callback for progress updates.

    Returns:
        Tuple of (success, result). On success, result is list of PIL Images.
    """
    client = ComfyUIClient()

    success, result = client.generate_images(
        prompt,
        num_images=num_images,
        callback=callback,
    )

    if success and project_folder:
        # Save images to project folder
        for i, img in enumerate(result):
            img_path = get_project_path(project_folder, f"generated_{i:02d}.png")
            img.save(img_path)

    return success, result


def save_selected_image(
    image: Image.Image,
    project_folder: Path | str,
    filename: str = "selected_image.png",
) -> Path:
    """Save the user-selected image to the project folder.

    Args:
        image: The PIL Image to save.
        project_folder: The project folder path.
        filename: The filename for the saved image.

    Returns:
        Path to the saved image.
    """
    img_path = get_project_path(project_folder, filename)
    image.save(img_path)
    return img_path


def check_comfyui() -> tuple[bool, str]:
    """Check if ComfyUI is available.

    Returns:
        Tuple of (success, message).
    """
    client = ComfyUIClient()
    return client.check_connection()
