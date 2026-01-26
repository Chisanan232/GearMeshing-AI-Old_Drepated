"""Model provider package initialization.

This module provides a convenient facade for accessing model provider
functionality, mirroring the structure of the prompt provider package.
"""

from .base import ModelProvider

__all__ = [
    "ModelProvider",
]
