"""TripoSR local inference wrapper for image-to-3D conversion."""

from pathlib import Path
from typing import Any

# Lazy imports for optional dependencies
TRIPOSR_AVAILABLE = False
_triposr_model = None


def _check_triposr() -> bool:
    """Check if TripoSR is available."""
    global TRIPOSR_AVAILABLE
    try:
        import torch
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
            "TripoSR is not installed. Install with: "
            "pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
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

    return _triposr_model


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
                "TripoSR is not available. Install with: "
                "pip install git+https://github.com/VAST-AI-Research/TripoSR.git"
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
                callback("Processing image...")

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # Remove background if needed (simple approach - assume white/gray bg)
            # TripoSR works better with transparent or removed backgrounds
            image_np = np.array(image)

            if callback:
                callback("Generating 3D model...")

            # Run inference
            with torch.no_grad():
                scene_codes = model([image], device=device)
                meshes = model.extract_mesh(scene_codes)

            if not meshes or len(meshes) == 0:
                return False, "TripoSR failed to generate mesh"

            mesh = meshes[0]

            if callback:
                callback("Exporting mesh...")

            # Convert to trimesh and export as GLB
            # TripoSR returns vertices and faces
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                tri_mesh = trimesh.Trimesh(
                    vertices=mesh.vertices.cpu().numpy(),
                    faces=mesh.faces.cpu().numpy(),
                )
            else:
                # If mesh is already a trimesh object
                tri_mesh = mesh

            glb_data = tri_mesh.export(file_type="glb")

            return True, glb_data

        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except torch.cuda.OutOfMemoryError:
            return False, "GPU out of memory. Try with a smaller image or use Meshy API."
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
