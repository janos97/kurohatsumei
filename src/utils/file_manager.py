"""File and project folder management utilities."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import get_config


def create_project_folder(base_name: str | None = None) -> Path:
    """Create a timestamped project folder.

    Args:
        base_name: Optional base name for the folder. If not provided,
                   uses just the timestamp.

    Returns:
        Path to the created project folder.
    """
    config = get_config()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if base_name:
        # Sanitize base name
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)
        safe_name = safe_name[:50]  # Limit length
        folder_name = f"{timestamp}_{safe_name}"
    else:
        folder_name = timestamp

    project_path = output_dir / folder_name
    project_path.mkdir(parents=True, exist_ok=True)

    return project_path


def get_project_path(project_folder: Path | str, filename: str) -> Path:
    """Get the full path for a file in a project folder.

    Args:
        project_folder: The project folder path.
        filename: The filename to append.

    Returns:
        Full path to the file.
    """
    return Path(project_folder) / filename


def save_json(project_folder: Path | str, filename: str, data: Any) -> Path:
    """Save data as JSON in the project folder.

    Args:
        project_folder: The project folder path.
        filename: The filename (should end with .json).
        data: Data to serialize as JSON.

    Returns:
        Path to the saved file.
    """
    filepath = get_project_path(project_folder, filename)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return filepath


def load_json(project_folder: Path | str, filename: str) -> Any:
    """Load JSON data from the project folder.

    Args:
        project_folder: The project folder path.
        filename: The filename to load.

    Returns:
        Parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    filepath = get_project_path(project_folder, filename)
    with open(filepath) as f:
        return json.load(f)


def save_image(project_folder: Path | str, filename: str, image_data: bytes) -> Path:
    """Save image data to the project folder.

    Args:
        project_folder: The project folder path.
        filename: The filename for the image.
        image_data: Raw image bytes.

    Returns:
        Path to the saved image.
    """
    filepath = get_project_path(project_folder, filename)
    with open(filepath, "wb") as f:
        f.write(image_data)
    return filepath


def list_project_files(project_folder: Path | str, pattern: str = "*") -> list[Path]:
    """List files in a project folder matching a pattern.

    Args:
        project_folder: The project folder path.
        pattern: Glob pattern to match files.

    Returns:
        List of matching file paths.
    """
    return list(Path(project_folder).glob(pattern))
