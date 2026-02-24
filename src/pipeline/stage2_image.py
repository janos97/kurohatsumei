"""Stage 2: Image generation with backend routing."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from PIL import Image

from ..config import get_config
from ..services.comfyui import ComfyUIClient
from ..utils.file_manager import get_project_path


# Backend type
ImageBackend = Literal["sd_cpp", "comfyui"]


def get_available_image_backends() -> dict[str, tuple[bool, str]]:
    """Get availability status of all image backends.

    Returns:
        Dict mapping backend name to (available, message) tuple.
    """
    backends = {}

    # Check sd_cpp
    try:
        from ..services.sd_cpp import StableDiffusionCppClient
        client = StableDiffusionCppClient()
        backends["sd_cpp"] = client.check_connection()
    except ImportError:
        backends["sd_cpp"] = (False, "stable-diffusion-cpp-python not installed")

    # Check ComfyUI
    client = ComfyUIClient()
    backends["comfyui"] = client.check_connection()

    return backends


def generate_images(
    prompt: str,
    project_folder: Path | str | None = None,
    num_images: int | None = None,
    backend: ImageBackend | None = None,
    callback: callable | None = None,
) -> tuple[bool, list[Image.Image] | str]:
    """Generate images from a text prompt.

    Args:
        prompt: The text prompt for image generation.
        project_folder: Optional folder to save generated images.
        num_images: Number of images to generate.
        backend: Which backend to use (sd_cpp, comfyui).
                 If None, uses default from config.
        callback: Optional callback for progress updates.

    Returns:
        Tuple of (success, result). On success, result is list of PIL Images.
    """
    config = get_config()

    if backend is None:
        backend = config.default_image_backend

    success, result = _generate_with_backend(prompt, backend, num_images, callback)

    if success and project_folder:
        # Save images to project folder
        for i, img in enumerate(result):
            img_path = get_project_path(project_folder, f"generated_{i:02d}.png")
            img.save(img_path)

    return success, result


def _generate_with_backend(
    prompt: str,
    backend: str,
    num_images: int | None = None,
    callback: callable | None = None,
) -> tuple[bool, list[Image.Image] | str]:
    """Generate images with a specific backend.

    Args:
        prompt: The text prompt.
        backend: Backend name.
        num_images: Number of images to generate.
        callback: Optional callback for progress updates.

    Returns:
        Tuple of (success, result).
    """
    if backend == "sd_cpp":
        try:
            from ..services.sd_cpp import StableDiffusionCppClient
            client = StableDiffusionCppClient()
            if not client.is_available():
                return False, "stable-diffusion-cpp-python not available"
            return client.generate_images(prompt, num_images=num_images, callback=callback)
        except ImportError:
            return False, "stable-diffusion-cpp-python not installed"

    elif backend == "comfyui":
        client = ComfyUIClient()
        return client.generate_images(prompt, num_images=num_images, callback=callback)

    else:
        return False, f"Unknown image backend: {backend}"


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


def check_image_backend() -> tuple[bool, str]:
    """Check if the default image backend is available.

    Returns:
        Tuple of (success, message).
    """
    config = get_config()
    backend = config.default_image_backend
    backends = get_available_image_backends()
    if backend in backends:
        return backends[backend]
    return False, f"Unknown default image backend: {backend}"
