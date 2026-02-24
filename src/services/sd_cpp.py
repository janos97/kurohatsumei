"""Stable Diffusion C++ local inference wrapper for text-to-image generation."""
from __future__ import annotations

import random
import time
from typing import Any

from PIL import Image

from ..config import get_config

# Lazy imports for optional dependencies
SD_CPP_AVAILABLE = False
_sd_model = None


def _check_sd_cpp() -> bool:
    """Check if stable-diffusion-cpp-python is available."""
    global SD_CPP_AVAILABLE
    try:
        import stable_diffusion_cpp
        SD_CPP_AVAILABLE = True
        return True
    except ImportError:
        SD_CPP_AVAILABLE = False
        return False


def _load_model():
    """Lazy-load the Stable Diffusion model."""
    global _sd_model

    if _sd_model is not None:
        return _sd_model

    if not _check_sd_cpp():
        raise ImportError(
            "stable-diffusion-cpp-python is not installed. Install with: "
            "pip install stable-diffusion-cpp-python"
        )

    from pathlib import Path
    import stable_diffusion_cpp

    config = get_config()
    model_path = config.sd_cpp_model_path

    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"SD model not found at {model_path}. Download a GGUF model "
            "and set sd_cpp.model_path in config.yaml"
        )

    _sd_model = stable_diffusion_cpp.StableDiffusion(
        model_path=model_path,
        n_threads=config.sd_cpp_n_threads,
        wtype=config.sd_cpp_wtype,
    )

    return _sd_model


class StableDiffusionCppClient:
    """Client for local Stable Diffusion inference via stable-diffusion-cpp-python."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if stable-diffusion-cpp-python is available."""
        if self._available is None:
            self._available = _check_sd_cpp()
        return self._available

    def generate_images(
        self,
        prompt: str,
        num_images: int | None = None,
        callback: callable | None = None,
    ) -> tuple[bool, list[Image.Image] | str]:
        """Generate images from a text prompt using Stable Diffusion C++.

        Args:
            prompt: The text prompt for image generation.
            num_images: Number of images to generate. Defaults to config value.
            callback: Optional callback(msg) for progress updates.

        Returns:
            Tuple of (success, result). On success, result is list of PIL Images.
        """
        if not self.is_available():
            return False, (
                "stable-diffusion-cpp-python is not installed. Install with: "
                "pip install stable-diffusion-cpp-python"
            )

        config = get_config()
        if num_images is None:
            num_images = config.sd_cpp_num_images

        try:
            if callback:
                callback("Loading Stable Diffusion model...")

            model = _load_model()

            seed = random.randint(0, 2**31 - 1)

            if callback:
                callback(f"Generating {num_images} image(s) (seed={seed})...")

            start_time = time.time()

            def step_callback(step: int, steps: int, t: float) -> bool:
                if callback:
                    elapsed = time.time() - start_time
                    callback(f"Step {step}/{steps} ({elapsed:.1f}s)")
                return True

            output = model.generate_image(
                prompt=prompt,
                width=config.sd_cpp_width,
                height=config.sd_cpp_height,
                cfg_scale=config.sd_cpp_cfg_scale,
                sample_steps=config.sd_cpp_sample_steps,
                sample_method=config.sd_cpp_sample_method,
                seed=seed,
                batch_count=num_images,
                progress_callback=step_callback,
            )

            return True, output

        except FileNotFoundError as e:
            return False, str(e)
        except Exception as e:
            return False, f"SD.cpp inference failed: {str(e)}"

    def check_connection(self) -> tuple[bool, str]:
        """Check if stable-diffusion-cpp-python is available and model exists.

        Returns:
            Tuple of (available, message).
        """
        if not self.is_available():
            return False, "stable-diffusion-cpp-python not installed"

        from pathlib import Path
        config = get_config()
        model_path = config.sd_cpp_model_path

        if not Path(model_path).exists():
            return False, f"Model file not found: {model_path}"

        return True, f"sd.cpp ready (model: {Path(model_path).name})"
