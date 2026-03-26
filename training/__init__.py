"""
training — Xybernetex AI OpenEngine training package.

Exports the pluggable reward interface, default reward implementation,
reward configuration, and the transition data container used by the
PPO trainer and rollout generator.
"""

from training.reward import (
    DefaultRewardFunction,
    RewardConfig,
    RewardFunction,
    TransitionData,
    get_default_reward_function,
)

__all__ = [
    "RewardFunction",
    "RewardConfig",
    "DefaultRewardFunction",
    "TransitionData",
    "get_default_reward_function",
]
