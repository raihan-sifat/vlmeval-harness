"""Model adapters.

Each module is imported lazily by the registry, so a user evaluating only local
models does not need the cloud provider SDKs installed and vice versa.
"""

from .base import ModelAdapter, encode_image, load_pil
from .echo import EchoModel

__all__ = ["ModelAdapter", "EchoModel", "encode_image", "load_pil"]
