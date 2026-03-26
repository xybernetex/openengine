"""
engine — Xybernetex AI OpenEngine core engine package.

This package contains the runtime components that execute goals:

- ``inference_worker``: top-level entry point; runs as a Redis consumer
  (queue mode) or a CLI single-shot executor (local dev mode).
- ``live_executor``: orchestrates a single goal episode — dispatches
  primitive actions, calls the LLM for HOW decisions, and advances state.
- ``llm_client``: thin provider-agnostic LLM client supporting Cloudflare
  Workers AI, OpenAI, Mistral, Anthropic, and Gemini.
- ``rl_chassis``: the ActorCritic policy network plus state vectorisation and
  action space definitions.
- ``xyber_env``: OpenAI-Gym-compatible environment wrapping the WorldEngine
  and WorkingMemory for PPO training.
- ``tick_logger``: structured per-step logging for replay and debugging.
- ``terminal``: coloured terminal output helpers (non-essential for headless).
"""

from engine.inference_worker import run_local_goal
from engine.live_executor import LiveExecutor
from engine.llm_client import LLMClient
from engine.rl_chassis import (
    ACTION_DIM,
    STATE_DIM,
    ActionSpace,
    ActorCritic,
    StateVectorizer,
    TransitionInfo,
)

__all__ = [
    # Entry points
    "run_local_goal",
    # Core classes
    "LiveExecutor",
    "LLMClient",
    # RL chassis
    "ActorCritic",
    "ActionSpace",
    "StateVectorizer",
    "TransitionInfo",
    "STATE_DIM",
    "ACTION_DIM",
]
