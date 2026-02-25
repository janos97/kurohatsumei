"""TripoSR local inference wrapper for image-to-3D conversion."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Lazy imports for optional dependencies
TRIPOSR_AVAILABLE = False
_triposr_model = None

# Path to vendored TripoSR source (no setup.py, tsr/ is a plain module)
_VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "TripoSR")


def _check_triposr() -> bool:
    """Check if TripoSR is available."""
    global TRIPOSR_AVAILABLE
    try:
        import torch

        # Ensure vendored TripoSR is on sys.path
        if _VENDOR_DIR not in sys.path:
            sys.path.insert(0, _VENDOR_DIR)

        import tsr  # TripoSR package
        TRIPOSR_AVAILABLE = True
        return True
    except ImportError:
        TRIPOSR_AVAILABLE = False
        return False


def _load_model():
    """Lazy-load the TripoSR model."""
    global _triposr_model

    if _triposr_model is not None:
        return _triposr_model

    if not _check_triposr():
        raise ImportError(
            "TripoSR is not installed. Clone into vendor/TripoSR and install deps."
        )

    import torch
    from tsr.system import TSR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _triposr_model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    _triposr_model.to(device)

    # Set chunk size for CPU memory efficiency
    _triposr_model.renderer.set_chunk_size(8192)

    return _triposr_model


def _remove_background(image):
    """Remove background from image using rembg, composite onto gray."""
    import numpy as np

    try:
        from rembg import remove
        rgba = remove(image)
    except ImportError:
        rgba = image.convert("RGBA")

    # Composite RGBA onto 50% gray background (matches TripoSR's run.py)
    from PIL import Image
    arr = np.array(rgba).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1 - arr[:, :, 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).astype(np.uint8))


class TripoSRClient:
    """Client for local TripoSR inference."""

    def __init__(self):
        self.model = None
        self._available = None

    def is_available(self) -> bool:
        """Check if TripoSR is available."""
        if self._available is None:
            self._available = _check_triposr()
        return self._available

    def image_to_3d(
        self,
        image_path: str | Path,
        callback: callable | None = None,
    ) -> tuple[bool, bytes | str]:
        """Convert an image to a 3D model using TripoSR.

        Args:
            image_path: Path to the input image.
            callback: Optional callback for progress updates.

        Returns:
            Tuple of (success, result). On success, result is GLB file bytes.
        """
        if not self.is_available():
            return False, (
                "TripoSR is not available. Clone into vendor/TripoSR and install deps."
            )

        try:
            from PIL import Image
            import numpy as np
            import torch
            import trimesh

            if callback:
                callback("Loading TripoSR model...")

            model = _load_model()
            device = next(model.parameters()).device

            if callback:
                callback("Removing background...")

            # Load and preprocess image with background removal
            image = Image.open(image_path).convert("RGB")
            image = _remove_background(image)

            if callback:
                callback("Generating 3D model (this may take several minutes on CPU)...")

            # Run inference
            with torch.no_grad():
                scene_codes = model([image], device=device)
                meshes = model.extract_mesh(
                    scene_codes,
                    has_vertex_color=True,
                    resolution=256,
                )

            if not meshes or len(meshes) == 0:
                return False, "TripoSR failed to generate mesh"

            mesh = meshes[0]

            if callback:
                callback("Exporting mesh...")

            # TripoSR returns trimesh.Trimesh objects directly
            if isinstance(mesh, trimesh.Trimesh):
                tri_mesh = mesh
            elif hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                tri_mesh = trimesh.Trimesh(
                    vertices=np.asarray(mesh.vertices),
                    faces=np.asarray(mesh.faces),
                )
            else:
                return False, "Unexpected mesh format from TripoSR"

            glb_data = tri_mesh.export(file_type="glb")

            return True, glb_data

        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except MemoryError:
            return False, "Out of memory. Try with a smaller image or use DepthMesh backend."
        except Exception as e:
            return False, f"TripoSR inference failed: {str(e)}"

    def check_availability(self) -> tuple[bool, str]:
        """Check if TripoSR is available and working.

        Returns:
            Tuple of (available, message).
        """
        if not self.is_available():
            return False, "TripoSR not installed"

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                return True, f"TripoSR available (GPU: {gpu_name})"
            else:
                return True, "TripoSR available (CPU mode - will be slow)"
        except Exception as e:
            return False, f"TripoSR check failed: {str(e)}"
