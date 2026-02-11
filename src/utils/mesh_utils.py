"""Mesh processing utilities using trimesh and pymeshfix."""

from pathlib import Path
from typing import Any

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    trimesh = None

try:
    import pymeshfix
    PYMESHFIX_AVAILABLE = True
except ImportError:
    PYMESHFIX_AVAILABLE = False
    pymeshfix = None


def check_trimesh() -> None:
    """Check if trimesh is available."""
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required for mesh operations. Install with: pip install trimesh")


def load_mesh(filepath: str | Path) -> "trimesh.Trimesh":
    """Load a mesh from file.

    Args:
        filepath: Path to the mesh file (STL, OBJ, GLB, PLY, etc.)

    Returns:
        Loaded trimesh object.
    """
    check_trimesh()
    mesh = trimesh.load(str(filepath), force="mesh")

    # If we got a scene, extract the geometry
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError("No geometry found in mesh file")
        # Combine all geometries
        meshes = list(mesh.geometry.values())
        if len(meshes) == 1:
            mesh = meshes[0]
        else:
            mesh = trimesh.util.concatenate(meshes)

    return mesh


def save_mesh(mesh: "trimesh.Trimesh", filepath: str | Path) -> Path:
    """Save a mesh to file.

    Args:
        mesh: The trimesh object to save.
        filepath: Output path. Format determined by extension.

    Returns:
        Path to the saved file.
    """
    filepath = Path(filepath)
    mesh.export(str(filepath))
    return filepath


def repair_mesh(
    mesh: "trimesh.Trimesh",
    close_boundaries: bool = False,
) -> "trimesh.Trimesh":
    """Repair a mesh for 3D printing.

    Always performs:
    - Fix inverted normals
    - Remove degenerate faces
    - Remove duplicate faces
    - Keep only largest connected component

    Optionally:
    - Close open boundaries (using pymeshfix)

    Args:
        mesh: Input mesh.
        close_boundaries: Whether to close open boundaries.

    Returns:
        Repaired mesh.
    """
    check_trimesh()

    # Work on a copy
    mesh = mesh.copy()

    # Fix inverted normals
    mesh.fix_normals()

    # Remove degenerate faces (zero area)
    mesh.remove_degenerate_faces()

    # Remove duplicate faces
    mesh.remove_duplicate_faces()

    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()

    # Keep only the largest connected component (removes debris)
    if hasattr(mesh, "split") and len(mesh.faces) > 0:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            # Find largest by face count
            mesh = max(components, key=lambda m: len(m.faces))

    # Close open boundaries if requested (requires pymeshfix)
    if close_boundaries:
        if not PYMESHFIX_AVAILABLE:
            raise ImportError("pymeshfix is required to close boundaries. Install with: pip install pymeshfix")

        # pymeshfix works with vertices and faces arrays
        mfix = pymeshfix.MeshFix(mesh.vertices, mesh.faces)
        mfix.repair(verbose=False)
        mesh = trimesh.Trimesh(vertices=mfix.v, faces=mfix.f)

    return mesh


def get_mesh_stats(mesh: "trimesh.Trimesh") -> dict[str, Any]:
    """Get statistics about a mesh.

    Args:
        mesh: The mesh to analyze.

    Returns:
        Dictionary with mesh statistics.
    """
    check_trimesh()

    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]

    # Check for open boundaries (edges belonging to only one face)
    edges = mesh.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    open_edges = np.sum(counts == 1)

    stats = {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.faces),
        "is_watertight": mesh.is_watertight,
        "open_edges": int(open_edges),
        "has_open_boundaries": open_edges > 0,
        "dimensions_mm": {
            "x": float(dimensions[0]),
            "y": float(dimensions[1]),
            "z": float(dimensions[2]),
        },
        "volume_mm3": float(mesh.volume) if mesh.is_watertight else None,
        "surface_area_mm2": float(mesh.area),
        "center": {
            "x": float(mesh.centroid[0]),
            "y": float(mesh.centroid[1]),
            "z": float(mesh.centroid[2]),
        },
    }

    return stats


def scale_to_fit_build_volume(
    mesh: "trimesh.Trimesh",
    max_x: float,
    max_y: float,
    max_z: float,
    margin: float = 5.0,
) -> "trimesh.Trimesh":
    """Scale mesh to fit within build volume while maintaining aspect ratio.

    Args:
        mesh: Input mesh.
        max_x: Maximum X dimension (mm).
        max_y: Maximum Y dimension (mm).
        max_z: Maximum Z dimension (mm).
        margin: Safety margin from edges (mm).

    Returns:
        Scaled mesh.
    """
    check_trimesh()

    # Apply margin
    available_x = max_x - 2 * margin
    available_y = max_y - 2 * margin
    available_z = max_z - margin  # Only margin at top

    # Get current dimensions
    bounds = mesh.bounds
    current_dims = bounds[1] - bounds[0]

    # Calculate scale factor (uniform scaling to maintain aspect ratio)
    scale_factors = [
        available_x / current_dims[0] if current_dims[0] > 0 else float("inf"),
        available_y / current_dims[1] if current_dims[1] > 0 else float("inf"),
        available_z / current_dims[2] if current_dims[2] > 0 else float("inf"),
    ]
    scale = min(scale_factors)

    # Only scale down, not up
    if scale < 1.0:
        mesh = mesh.copy()
        mesh.apply_scale(scale)

    return mesh


def center_on_build_plate(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Center mesh on build plate (XY center, Z bottom at 0).

    Args:
        mesh: Input mesh.

    Returns:
        Centered mesh.
    """
    check_trimesh()

    mesh = mesh.copy()

    # Get bounds
    bounds = mesh.bounds

    # Calculate translation to center XY and place bottom at Z=0
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    center_y = (bounds[0][1] + bounds[1][1]) / 2
    bottom_z = bounds[0][2]

    translation = [-center_x, -center_y, -bottom_z]
    mesh.apply_translation(translation)

    return mesh


def mesh_to_glb_bytes(mesh: "trimesh.Trimesh") -> bytes:
    """Export mesh to GLB format bytes for Gradio Model3D viewer.

    Args:
        mesh: The mesh to export.

    Returns:
        GLB file contents as bytes.
    """
    check_trimesh()
    return mesh.export(file_type="glb")


def mesh_to_stl_bytes(mesh: "trimesh.Trimesh") -> bytes:
    """Export mesh to binary STL format bytes.

    Args:
        mesh: The mesh to export.

    Returns:
        Binary STL file contents.
    """
    check_trimesh()
    return mesh.export(file_type="stl")
