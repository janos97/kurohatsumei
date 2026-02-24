"""TRELLIS local inference wrapper for image-to-3D conversion."""
from __future__ import annotations

from pathlib import Path
from typing import Any

# Lazy imports for optional dependencies
TRELLIS_AVAILABLE = False
_trellis_pipeline = None


def _check_trellis() -> bool:
    """Check if TRELLIS is available."""
    global TRELLIS_AVAILABLE
    try:
        import torch
        from trellis.pipelines import TrellisImageTo3DPipeline
        TRELLIS_AVAILABLE = True
        return True
    except ImportError:
        TRELLIS_AVAILABLE = False
        return False


def _load_pipeline():
    """Lazy-load the TRELLIS pipeline."""
    global _trellis_pipeline

    if _trellis_pipeline is not None:
        return _trellis_pipeline

    if not _check_trellis():
        raise ImportError(
            "TRELLIS is not installed. See: https://github.com/microsoft/TRELLIS"
        )

    import torch
    from trellis.pipelines import TrellisImageTo3DPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(
        "microsoft/TRELLIS-image-large"
    )
    _trellis_pipeline.to(device)

    return _trellis_pipeline


class TrellisClient:
    """Client for local TRELLIS inference."""

    def __init__(self):
        self.pipeline = None
        self._available = None

    def is_available(self) -> bool:
        """Check if TRELLIS is available."""
        if self._available is None:
            self._available = _check_trellis()
        return self._available

    def image_to_3d(
        self,
        image_path: str | Path,
        callback: callable | None = None,
    ) -> tuple[bool, bytes | str]:
        """Convert an image to a 3D model using TRELLIS.

        Args:
            image_path: Path to the input image.
            callback: Optional callback for progress updates.

        Returns:
            Tuple of (success, result). On success, result is GLB file bytes.
        """
        if not self.is_available():
            return False, (
                "TRELLIS is not available. See: https://github.com/microsoft/TRELLIS"
            )

        try:
            from PIL import Image
            import torch
            import trimesh

            if callback:
                callback("Loading TRELLIS pipeline...")

            pipeline = _load_pipeline()

            if callback:
                callback("Processing image...")

            # Load image
            image = Image.open(image_path).convert("RGB")

            if callback:
                callback("Generating 3D model (this may take a while)...")

            # Run inference
            with torch.no_grad():
                outputs = pipeline(
                    image,
                    seed=42,
                )

            # Extract mesh from outputs
            # TRELLIS outputs include 3D Gaussians and mesh
            if hasattr(outputs, "mesh") and outputs.mesh is not None:
                mesh = outputs.mesh
            elif hasattr(outputs, "extract_mesh"):
                mesh = outputs.extract_mesh()
            else:
                return False, "TRELLIS did not produce a mesh output"

            if callback:
                callback("Exporting mesh...")

            # Convert to trimesh format
            if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
                vertices = mesh.vertices
                faces = mesh.faces

                # Handle torch tensors
                if hasattr(vertices, "cpu"):
                    vertices = vertices.cpu().numpy()
                if hasattr(faces, "cpu"):
                    faces = faces.cpu().numpy()

                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                tri_mesh = mesh

            glb_data = tri_mesh.export(file_type="glb")

            return True, glb_data

        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except torch.cuda.OutOfMemoryError:
            return False, "GPU out of memory. Try with a smaller image or use Meshy API."
        except Exception as e:
            return False, f"TRELLIS inference failed: {str(e)}"

    def check_availability(self) -> tuple[bool, str]:
        """Check if TRELLIS is available and working.

        Returns:
            Tuple of (available, message).
        """
        if not self.is_available():
            return False, "TRELLIS not installed"

        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if vram < 12:
                    return True, f"TRELLIS available (GPU: {gpu_name}, VRAM: {vram:.1f}GB - may be insufficient)"
                return True, f"TRELLIS available (GPU: {gpu_name}, VRAM: {vram:.1f}GB)"
            else:
                return False, "TRELLIS requires CUDA GPU"
        except Exception as e:
            return False, f"TRELLIS check failed: {str(e)}"
