"""
core — Xybernetex AI OpenEngine foundational components.

This package contains the stateless building blocks that the engine uses
during execution:

- ``world_engine``: maintains the structured world model — goal aspects,
  artefacts, tool results, and overall task progress.
- ``working_memory``: the scratchpad updated each step; provides the raw
  signals that are vectorised into the policy's observation tensor.
- ``code_executor``: sandboxed Python code execution with timeout and
  output capture.
- ``web_tools``: lightweight web search and fetch helpers (Tavily / raw HTTP).
"""

from core.working_memory import WorkingMemory
from core.world_engine import Archivist, Integrator, WorldEngine

__all__ = [
    "WorkingMemory",
    "WorldEngine",
    "Archivist",
    "Integrator",
]
