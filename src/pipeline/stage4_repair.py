"""Stage 4: Mesh repair and STL export."""

from pathlib import Path
from typing import Any

from ..config import get_config
from ..utils.file_manager import get_project_path
from ..utils.mesh_utils import (
    load_mesh,
    save_mesh,
    repair_mesh,
    get_mesh_stats,
    scale_to_fit_build_volume,
    center_on_build_plate,
    mesh_to_glb_bytes,
    mesh_to_stl_bytes,
)


def repair_and_export(
    mesh_path: str | Path | None = None,
    mesh_bytes: bytes | None = None,
    project_folder: Path | str | None = None,
    close_boundaries: bool = False,
) -> tuple[bool, dict[str, Any] | str]:
    """Repair a mesh and prepare it for 3D printing.

    Args:
        mesh_path: Path to the input mesh file.
        mesh_bytes: Raw mesh bytes (GLB format). Used if mesh_path is None.
        project_folder: Optional folder to save repaired mesh.
        close_boundaries: Whether to close open boundaries (uses pymeshfix).

    Returns:
        Tuple of (success, result). On success, result is dict with:
        - 'stats': Mesh statistics
        - 'glb_bytes': GLB file bytes for viewer
        - 'stl_bytes': STL file bytes for printing
        - 'stl_path': Path to saved STL (if project_folder provided)
    """
    config = get_config()
    build_vol = config.build_volume

    try:
        # Load the mesh
        if mesh_path:
            mesh = load_mesh(mesh_path)
        elif mesh_bytes:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
                f.write(mesh_bytes)
                temp_path = f.name
            mesh = load_mesh(temp_path)
            Path(temp_path).unlink()  # Clean up temp file
        else:
            return False, "Either mesh_path or mesh_bytes must be provided"

        # Repair the mesh
        mesh = repair_mesh(mesh, close_boundaries=close_boundaries)

        # Scale to fit build volume
        mesh = scale_to_fit_build_volume(
            mesh,
            max_x=build_vol["x"],
            max_y=build_vol["y"],
            max_z=build_vol["z"],
            margin=build_vol["margin"],
        )

        # Center on build plate
        mesh = center_on_build_plate(mesh)

        # Get stats
        stats = get_mesh_stats(mesh)

        # Export formats
        glb_bytes = mesh_to_glb_bytes(mesh)
        stl_bytes = mesh_to_stl_bytes(mesh)

        result = {
            "stats": stats,
            "glb_bytes": glb_bytes,
            "stl_bytes": stl_bytes,
            "stl_path": None,
        }

        # Save to project folder if provided
        if project_folder:
            glb_path = get_project_path(project_folder, "repaired_mesh.glb")
            stl_path = get_project_path(project_folder, "final_print.stl")

            with open(glb_path, "wb") as f:
                f.write(glb_bytes)

            with open(stl_path, "wb") as f:
                f.write(stl_bytes)

            result["stl_path"] = stl_path

        return True, result

    except ImportError as e:
        return False, f"Missing dependency: {str(e)}"
    except Exception as e:
        return False, f"Mesh repair failed: {str(e)}"


def format_stats_table(stats: dict[str, Any]) -> str:
    """Format mesh stats as a readable table.

    Args:
        stats: Stats dictionary from get_mesh_stats.

    Returns:
        Formatted string table.
    """
    dims = stats["dimensions_mm"]
    center = stats["center"]

    lines = [
        "| Property | Value |",
        "|----------|-------|",
        f"| Vertices | {stats['vertices']:,} |",
        f"| Faces | {stats['faces']:,} |",
        f"| Watertight | {'Yes' if stats['is_watertight'] else 'No'} |",
        f"| Open Edges | {stats['open_edges']} |",
        f"| Dimensions (mm) | {dims['x']:.1f} x {dims['y']:.1f} x {dims['z']:.1f} |",
        f"| Surface Area (mm²) | {stats['surface_area_mm2']:.1f} |",
    ]

    if stats["volume_mm3"] is not None:
        lines.append(f"| Volume (mm³) | {stats['volume_mm3']:.1f} |")
    else:
        lines.append("| Volume | N/A (not watertight) |")

    return "\n".join(lines)
