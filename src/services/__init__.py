"""Service clients for external APIs and local inference."""

from .ollama import OllamaClient
from .comfyui import ComfyUIClient
from .meshy import MeshyClient

__all__ = [
    "OllamaClient",
    "ComfyUIClient",
    "MeshyClient",
]

# Optional imports for local inference
try:
    from .triposr import TripoSRClient
    __all__.append("TripoSRClient")
except ImportError:
    TripoSRClient = None

try:
    from .trellis import TrellisClient
    __all__.append("TrellisClient")
except ImportError:
    TrellisClient = None
