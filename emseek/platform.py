"""Compatibility shim for the Platform module.

This module re-exports the core Platform implementation from
`emseek.core.platform` to keep public imports stable::

    from emseek.platform import Platform

No functionality is changed.
"""

from .core.platform import Platform  # noqa: F401

__all__ = ['Platform']

