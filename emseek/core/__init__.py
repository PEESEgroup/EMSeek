"""Core platform modules for EMSeek.

This package hosts the platform runtime and shared protocol helpers.
Agent packages should not depend on CLI or app entrypoints, only on
these stable core interfaces.
"""

# Re-export key symbols for convenience
from .platform import Platform  # noqa: F401
from .protocol import (
    normalize_mm_request,
    build_stream_step,
    build_stream_final,
    save_base64_image,
    path_to_base64_image,
    to_image_refs,
)  # noqa: F401

__all__ = [
    'Platform',
    'normalize_mm_request',
    'build_stream_step',
    'build_stream_final',
    'save_base64_image',
    'path_to_base64_image',
    'to_image_refs',
]

