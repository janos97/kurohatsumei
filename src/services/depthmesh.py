"""DepthMesh: Fast 2.5D relief mesh generation from depth estimation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Lazy imports for optional dependencies
DEPTHMESH_AVAILABLE = False
_depth_pipeline = None


def _check_depthmesh() -> bool:
    """Check if DepthMesh dependencies are available."""
    global DEPTHMESH_AVAILABLE
    try:
        import torch
        import transformers
        import trimesh
        DEPTHMESH_AVAILABLE = True
        return True
    except ImportError:
        DEPTHMESH_AVAILABLE = False
        return False


def _load_pipeline(model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    """Lazy-load the depth estimation pipeline."""
    global _depth_pipeline

    if _depth_pipeline is not None:
        return _depth_pipeline

    if not _check_depthmesh():
        raise ImportError("DepthMesh dependencies not installed.")

    from transformers import pipeline

    _depth_pipeline = pipeline(
        "depth-estimation",
        model=model_name,
        device="cpu",
    )

    return _depth_pipeline


class DepthMeshClient:
    """Client for depth-based 2.5D relief mesh generation."""

    def __init__(self):
        self._available = None

    def is_available(self) -> bool:
        """Check if DepthMesh is available."""
        if self._available is None:
            self._available = _check_depthmesh()
        return self._available

    def image_to_3d(
        self,
        image_path: str | Path,
        callback: callable | None = None,
    ) -> tuple[bool, bytes | str]:
        """Convert an image to a 2.5D relief mesh.

        Args:
            image_path: Path to the input image.
            callback: Optional callback for progress updates.

        Returns:
            Tuple of (success, result). On success, result is GLB file bytes.
        """
        if not self.is_available():
            return False, "DepthMesh not available. Install torch, transformers, trimesh."

        try:
            from PIL import Image
            import trimesh

            from ..config import get_config
            config = get_config()

            resolution = config.get("depthmesh", "resolution", default=256)
            depth_scale = config.get("depthmesh", "depth_scale", default=1.0)
            back_offset = config.get("depthmesh", "back_offset", default=0.1)
            edge_threshold = config.get("depthmesh", "edge_threshold", default=0.3)
            model_name = config.get(
                "depthmesh", "model",
                default="depth-anything/Depth-Anything-V2-Small-hf",
            )

            if callback:
                callback("Loading depth estimation model...")

            pipe = _load_pipeline(model_name)

            if callback:
                callback("Processing image...")

            # Load and resize image
            image = Image.open(image_path).convert("RGB")
            image = image.resize((resolution, resolution), Image.LANCZOS)
            img_array = np.array(image, dtype=np.float32) / 255.0

            if callback:
                callback("Estimating depth...")

            # Run depth estimation
            result = pipe(image)
            depth_map = np.array(result["depth"], dtype=np.float32)

            # Resize depth map to match resolution
            from PIL import Image as PILImage
            depth_pil = PILImage.fromarray(depth_map)
            depth_pil = depth_pil.resize((resolution, resolution), PILImage.LANCZOS)
            depth_map = np.array(depth_pil, dtype=np.float32)

            # Normalize depth to [0, depth_scale]
            d_min, d_max = depth_map.min(), depth_map.max()
            if d_max - d_min > 1e-6:
                depth_map = (depth_map - d_min) / (d_max - d_min) * depth_scale
            else:
                depth_map = np.zeros_like(depth_map)

            if callback:
                callback("Building mesh...")

            H, W = resolution, resolution

            # --- Front surface ---
            front_verts, front_colors, front_faces = _build_front_surface(
                depth_map, img_array, H, W, edge_threshold,
            )

            # --- Back plate ---
            back_verts, back_colors, back_faces = _build_back_plate(
                img_array, H, W, back_offset, len(front_verts),
            )

            # --- Side walls ---
            side_verts, side_colors, side_faces = _build_side_walls(
                depth_map, img_array, H, W, back_offset,
                len(front_verts) + len(back_verts),
            )

            # Assemble
            all_verts = np.concatenate([front_verts, back_verts, side_verts], axis=0)
            all_colors = np.concatenate([front_colors, back_colors, side_colors], axis=0)
            all_faces = np.concatenate([front_faces, back_faces, side_faces], axis=0)

            if callback:
                callback("Exporting mesh...")

            mesh = trimesh.Trimesh(
                vertices=all_verts,
                faces=all_faces,
                vertex_colors=(all_colors * 255).astype(np.uint8),
                process=False,
            )
            trimesh.repair.fix_normals(mesh)

            glb_data = mesh.export(file_type="glb")
            return True, glb_data

        except FileNotFoundError:
            return False, f"Image file not found: {image_path}"
        except MemoryError:
            return False, "Out of memory. Try a smaller resolution in config."
        except Exception as e:
            return False, f"DepthMesh generation failed: {str(e)}"

    def check_availability(self) -> tuple[bool, str]:
        """Check if DepthMesh is available and working."""
        if not self.is_available():
            return False, "DepthMesh not installed (need torch, transformers, trimesh)"
        return True, "DepthMesh available (CPU depth estimation)"


def _build_front_surface(
    depth_map: np.ndarray,
    img_array: np.ndarray,
    H: int,
    W: int,
    edge_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the front surface grid with depth displacement."""
    # Create vertex grid
    ys, xs = np.mgrid[0:H, 0:W]
    # Normalize to [-0.5, 0.5]
    x_coords = xs.astype(np.float32) / (W - 1) - 0.5
    y_coords = -(ys.astype(np.float32) / (H - 1) - 0.5)  # flip Y
    z_coords = depth_map

    verts = np.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=-1)
    colors = img_array.reshape(-1, 3)

    # Build faces (two triangles per quad), skipping large depth discontinuities
    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            i00 = r * W + c
            i01 = r * W + (c + 1)
            i10 = (r + 1) * W + c
            i11 = (r + 1) * W + (c + 1)

            d00 = depth_map[r, c]
            d01 = depth_map[r, c + 1]
            d10 = depth_map[r + 1, c]
            d11 = depth_map[r + 1, c + 1]

            depths = [d00, d01, d10, d11]
            max_diff = max(depths) - min(depths)

            if max_diff > edge_threshold:
                continue

            faces.append([i00, i10, i01])
            faces.append([i01, i10, i11])

    faces = np.array(faces, dtype=np.int64) if faces else np.empty((0, 3), dtype=np.int64)
    return verts, colors, faces


def _build_back_plate(
    img_array: np.ndarray,
    H: int,
    W: int,
    back_offset: float,
    vert_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a flat back plate."""
    ys, xs = np.mgrid[0:H, 0:W]
    x_coords = xs.astype(np.float32) / (W - 1) - 0.5
    y_coords = -(ys.astype(np.float32) / (H - 1) - 0.5)
    z_coords = np.full((H, W), -back_offset, dtype=np.float32)

    verts = np.stack([x_coords.ravel(), y_coords.ravel(), z_coords.ravel()], axis=-1)
    # Use a neutral gray for back plate
    colors = np.full((H * W, 3), 0.5, dtype=np.float32)

    faces = []
    for r in range(H - 1):
        for c in range(W - 1):
            i00 = vert_offset + r * W + c
            i01 = vert_offset + r * W + (c + 1)
            i10 = vert_offset + (r + 1) * W + c
            i11 = vert_offset + (r + 1) * W + (c + 1)
            # Reversed winding for back face
            faces.append([i00, i01, i10])
            faces.append([i01, i11, i10])

    faces = np.array(faces, dtype=np.int64)
    return verts, colors, faces


def _build_side_walls(
    depth_map: np.ndarray,
    img_array: np.ndarray,
    H: int,
    W: int,
    back_offset: float,
    vert_offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build side walls connecting front perimeter to back perimeter."""
    verts = []
    colors = []
    faces = []
    vi = 0  # local vertex index

    # Collect perimeter pixels in order: top, right, bottom (reversed), left (reversed)
    perimeter = []
    # Top row (left to right)
    for c in range(W):
        perimeter.append((0, c))
    # Right column (top to bottom, skip first)
    for r in range(1, H):
        perimeter.append((r, W - 1))
    # Bottom row (right to left, skip first)
    for c in range(W - 2, -1, -1):
        perimeter.append((H - 1, c))
    # Left column (bottom to top, skip first and last)
    for r in range(H - 2, 0, -1):
        perimeter.append((r, 0))

    for idx in range(len(perimeter)):
        r0, c0 = perimeter[idx]
        r1, c1 = perimeter[(idx + 1) % len(perimeter)]

        # Front edge vertices
        x0 = c0 / (W - 1) - 0.5
        y0 = -(r0 / (H - 1) - 0.5)
        z0_front = float(depth_map[r0, c0])
        z0_back = -back_offset

        x1 = c1 / (W - 1) - 0.5
        y1 = -(r1 / (H - 1) - 0.5)
        z1_front = float(depth_map[r1, c1])
        z1_back = -back_offset

        col0 = img_array[r0, c0]
        col1 = img_array[r1, c1]

        base = vert_offset + vi
        verts.extend([
            [x0, y0, z0_front],
            [x0, y0, z0_back],
            [x1, y1, z1_front],
            [x1, y1, z1_back],
        ])
        colors.extend([col0, col0, col1, col1])

        # Two triangles for the quad
        faces.append([base, base + 1, base + 2])
        faces.append([base + 2, base + 1, base + 3])
        vi += 4

    if not verts:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.int64),
        )

    verts = np.array(verts, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    faces = np.array(faces, dtype=np.int64)
    return verts, colors, faces
