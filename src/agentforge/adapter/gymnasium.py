"""Legacy compatibility module for GymnasiumAdapter.

Canonical implementation:
    agentforge.environment.gym_adapter.GymnasiumAdapter

This module remains import-compatible for legacy callers and does not define
a second GymnasiumAdapter implementation.
"""

from agentforge.environment.gym_adapter import GymnasiumAdapter

__all__ = ["GymnasiumAdapter"]
