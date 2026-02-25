"""Configuration loader for KuroHatsumei."""

import os
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = {
    "ollama": {
        "url": "http://localhost:11434",
        "model": "llama3.1:8b",
    },
    "comfyui": {
        "url": "http://localhost:8188",
        "workflow_path": "workflows/flux_txt2img.json",
        "num_images": 4,
        "poll_interval": 1.0,
    },
    "meshy": {
        "api_key": "",
        "api_url": "https://api.meshy.ai",
        "poll_interval": 5.0,
    },
    "sd_cpp": {
        "model_path": "models/sd15_q8_0.gguf",
        "n_threads": -1,
        "wtype": "default",
        "width": 512,
        "height": 512,
        "cfg_scale": 7.0,
        "sample_steps": 20,
        "sample_method": "euler_a",
        "num_images": 4,
    },
    "depthmesh": {
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "resolution": 256,
        "depth_scale": 1.0,
        "back_offset": 0.1,
        "edge_threshold": 0.3,
    },
    "build_volume": {
        "x": 223,
        "y": 126,
        "z": 230,
        "margin": 5,
    },
    "default_image_backend": "comfyui",
    "default_3d_backend": "triposr",
    "output_dir": "output",
}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class Config:
    """Configuration manager with environment variable override support."""

    def __init__(self, config_path: str | Path | None = None):
        self._config = DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_path:
            self._load_from_file(config_path)
        else:
            # Try default locations
            for path in ["config.yaml", "config.yml"]:
                if Path(path).exists():
                    self._load_from_file(path)
                    break

        # Apply environment variable overrides
        self._apply_env_overrides()

    def _load_from_file(self, path: str | Path) -> None:
        """Load configuration from YAML file."""
        path = Path(path)
        if path.exists():
            with open(path) as f:
                file_config = yaml.safe_load(f) or {}
            self._config = deep_merge(self._config, file_config)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            "OLLAMA_URL": ("ollama", "url"),
            "OLLAMA_MODEL": ("ollama", "model"),
            "COMFYUI_URL": ("comfyui", "url"),
            "MESHY_API_KEY": ("meshy", "api_key"),
            "KUROHATSUMEI_OUTPUT_DIR": ("output_dir",),
            "KUROHATSUMEI_IMAGE_BACKEND": ("default_image_backend",),
            "KUROHATSUMEI_3D_BACKEND": ("default_3d_backend",),
            "SD_CPP_MODEL_PATH": ("sd_cpp", "model_path"),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)

    def _set_nested(self, path: tuple[str, ...], value: Any) -> None:
        """Set a nested configuration value."""
        current = self._config
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value

    def get(self, *keys: str, default: Any = None) -> Any:
        """Get a configuration value by key path."""
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    @property
    def ollama_url(self) -> str:
        return self.get("ollama", "url")

    @property
    def ollama_model(self) -> str:
        return self.get("ollama", "model")

    @property
    def comfyui_url(self) -> str:
        return self.get("comfyui", "url")

    @property
    def comfyui_workflow_path(self) -> str:
        return self.get("comfyui", "workflow_path")

    @property
    def comfyui_num_images(self) -> int:
        return self.get("comfyui", "num_images")

    @property
    def comfyui_poll_interval(self) -> float:
        return self.get("comfyui", "poll_interval")

    @property
    def meshy_api_key(self) -> str:
        return self.get("meshy", "api_key")

    @property
    def meshy_api_url(self) -> str:
        return self.get("meshy", "api_url")

    @property
    def meshy_poll_interval(self) -> float:
        return self.get("meshy", "poll_interval")

    @property
    def sd_cpp_model_path(self) -> str:
        return self.get("sd_cpp", "model_path")

    @property
    def sd_cpp_n_threads(self) -> int:
        return self.get("sd_cpp", "n_threads")

    @property
    def sd_cpp_wtype(self) -> str:
        return self.get("sd_cpp", "wtype")

    @property
    def sd_cpp_width(self) -> int:
        return self.get("sd_cpp", "width")

    @property
    def sd_cpp_height(self) -> int:
        return self.get("sd_cpp", "height")

    @property
    def sd_cpp_cfg_scale(self) -> float:
        return self.get("sd_cpp", "cfg_scale")

    @property
    def sd_cpp_sample_steps(self) -> int:
        return self.get("sd_cpp", "sample_steps")

    @property
    def sd_cpp_sample_method(self) -> str:
        return self.get("sd_cpp", "sample_method")

    @property
    def sd_cpp_num_images(self) -> int:
        return self.get("sd_cpp", "num_images")

    @property
    def depthmesh_model(self) -> str:
        return self.get("depthmesh", "model")

    @property
    def depthmesh_resolution(self) -> int:
        return self.get("depthmesh", "resolution")

    @property
    def depthmesh_depth_scale(self) -> float:
        return self.get("depthmesh", "depth_scale")

    @property
    def depthmesh_back_offset(self) -> float:
        return self.get("depthmesh", "back_offset")

    @property
    def depthmesh_edge_threshold(self) -> float:
        return self.get("depthmesh", "edge_threshold")

    @property
    def default_image_backend(self) -> str:
        return self.get("default_image_backend")

    @property
    def build_volume(self) -> dict[str, int]:
        return self.get("build_volume")

    @property
    def default_3d_backend(self) -> str:
        return self.get("default_3d_backend")

    @property
    def output_dir(self) -> str:
        return self.get("output_dir")


# Global config instance
_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config(config_path: str | Path | None = None) -> Config:
    """Reload configuration from file."""
    global _config
    _config = Config(config_path)
    return _config
