"""Compatibility shim for protocol helpers.

Public functions are now implemented in `emseek.core.protocol`.
This module re-exports them to keep existing imports working::

    from emseek.protocol import normalize_agent_input, build_stream_step, ...

No functionality is changed.
"""

from .core.protocol import *  # noqa: F401,F403

