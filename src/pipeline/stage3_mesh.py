"""Stage 3: Image-to-3D mesh generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from ..config import get_config
from ..services.meshy import MeshyClient
from ..utils.file_manager import get_project_path


# Backend type
Backend = Literal["triposr", "trellis", "meshy", "depthmesh"]


def get_available_backends() -> dict[str, tuple[bool, str]]:
    """Get availability status of all 3D backends.

    Returns:
        Dict mapping backend name to (available, message) tuple.
    """
    backends = {}

    # Check TripoSR
    try:
        from ..services.triposr import TripoSRClient
        client = TripoSRClient()
        backends["triposr"] = client.check_availability()
    except ImportError:
        backends["triposr"] = (False, "TripoSR not installed")

    # Check TRELLIS
    try:
        from ..services.trellis import TrellisClient
        client = TrellisClient()
        backends["trellis"] = client.check_availability()
    except ImportError:
        backends["trellis"] = (False, "TRELLIS not installed")

    # Check DepthMesh
    try:
        from ..services.depthmesh import DepthMeshClient
        client = DepthMeshClient()
        backends["depthmesh"] = client.check_availability()
    except ImportError:
        backends["depthmesh"] = (False, "DepthMesh not installed")

    # Check Meshy
    client = MeshyClient()
    backends["meshy"] = client.check_connection()

    return backends


def generate_mesh(
    image_path: str | Path,
    project_folder: Path | str | None = None,
    backend: Backend | None = None,
    callback: callable | None = None,
) -> tuple[bool, bytes | str]:
    """Generate a 3D mesh from an image.

    Args:
        image_path: Path to the input image.
        project_folder: Optional folder to save the generated mesh.
        backend: Which backend to use (triposr, trellis, meshy).
                 If None, uses default from config with fallback.
        callback: Optional callback for progress updates.

    Returns:
        Tuple of (success, result). On success, result is GLB file bytes.
    """
    config = get_config()

    if backend is None:
        backend = config.default_3d_backend

    # Try the requested backend, with fallback
    backends_to_try = [backend]

    # Add fallback order
    fallback_order = ["triposr", "depthmesh", "trellis", "meshy"]
    for fb in fallback_order:
        if fb not in backends_to_try:
            backends_to_try.append(fb)

    last_error = "No backends available"

    for backend_name in backends_to_try:
        if callback:
            callback(f"Trying {backend_name}...")

        success, result = _generate_with_backend(
            image_path,
            backend_name,
            callback,
        )

        if success:
            if project_folder:
                # Save the raw mesh
                mesh_path = get_project_path(project_folder, "raw_mesh.glb")
                with open(mesh_path, "wb") as f:
                    f.write(result)

            return True, result
        else:
            last_error = result
            if callback:
                callback(f"{backend_name} failed: {result}")

            # If this was the explicitly requested backend, don't try others
            if backend_name == backend and backend_name != config.default_3d_backend:
                return False, result

    return False, f"All backends failed. Last error: {last_error}"


def _generate_with_backend(
    image_path: str | Path,
    backend: str,
    callback: callable | None = None,
) -> tuple[bool, bytes | str]:
    """Generate mesh with a specific backend.

    Args:
        image_path: Path to the input image.
        backend: Backend name.
        callback: Optional callback for progress updates.

    Returns:
        Tuple of (success, result).
    """
    if backend == "triposr":
        try:
            from ..services.triposr import TripoSRClient
            client = TripoSRClient()
            if not client.is_available():
                return False, "TripoSR not available"
            return client.image_to_3d(image_path, callback)
        except ImportError:
            return False, "TripoSR not installed"

    elif backend == "trellis":
        try:
            from ..services.trellis import TrellisClient
            client = TrellisClient()
            if not client.is_available():
                return False, "TRELLIS not available"
            return client.image_to_3d(image_path, callback)
        except ImportError:
            return False, "TRELLIS not installed"

    elif backend == "depthmesh":
        try:
            from ..services.depthmesh import DepthMeshClient
            client = DepthMeshClient()
            if not client.is_available():
                return False, "DepthMesh not available"
            return client.image_to_3d(image_path, callback)
        except ImportError:
            return False, "DepthMesh not installed"

    elif backend == "meshy":
        client = MeshyClient()
        return client.image_to_3d(image_path, callback)

    else:
        return False, f"Unknown backend: {backend}"
