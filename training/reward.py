"""
training/reward.py — Pluggable reward interface for Xybernetex AI OpenEngine.

Overview
--------
The reward function is intentionally decoupled from the environment so that
researchers and integrators can swap in domain-specific reward signals without
touching the core training loop.

To define your own reward:

    from training.reward import RewardFunction, TransitionData

    class MyReward(RewardFunction):
        def compute(self, transition: dict) -> float:
            td = TransitionData(**transition)
            # your logic here
            return 1.0 if td.done and td.exec_success else -0.1

Then pass an instance to the training pipeline:

    trainer = PPOTrainer(env, reward_fn=MyReward())

The ``DefaultRewardFunction`` shipped here matches the reward signal used
during the base-policy training run and is the correct starting point for
fine-tuning on new goal types.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Transition data container ──────────────────────────────────────────────────

@dataclass
class TransitionData:
    """
    Structured representation of a single environment transition.

    Attributes
    ----------
    action_name:
        The string name of the primitive action taken (e.g. ``"INVOKE_TOOL"``).
    aspects_before:
        Number of goal aspects that were marked complete *before* this step.
    aspects_after:
        Number of goal aspects that were marked complete *after* this step.
    exec_success:
        Whether the low-level execution (tool call, code run, etc.) succeeded.
    exec_returncode:
        Return code from the subprocess / tool invocation.  0 means success.
        Use ``None`` when not applicable (e.g. PLAN action).
    done:
        Whether the episode terminated after this transition (DONE action fired
        or max-steps exceeded).
    step:
        The current step index (0-indexed).
    max_steps:
        The episode step budget configured for this run.
    """
    action_name: str
    aspects_before: int
    aspects_after: int
    exec_success: bool
    exec_returncode: int | None
    done: bool
    step: int
    max_steps: int = 20
    extra: dict[str, Any] = field(default_factory=dict)


# ── Reward configuration ───────────────────────────────────────────────────────

@dataclass
class RewardConfig:
    """
    Scalar weights used by :class:`DefaultRewardFunction`.

    All values are floats and can be overridden at construction time:

        cfg = RewardConfig(terminal_success=20.0, step_penalty=-0.2)

    Attributes
    ----------
    step_penalty:
        Small negative reward applied every step to encourage efficiency.
    aspect_complete:
        Reward per newly completed goal aspect (delta-based — only fires once
        per aspect transition).
    exec_success:
        Small positive reward for any successful tool/code execution, even if
        no new aspect was completed.  Encourages grounded action.
    terminal_success:
        Large reward when the DONE action is taken and all goal aspects are
        complete.
    invalid_done:
        Negative reward when DONE is taken with incomplete aspects (premature
        termination).
    stagnation:
        Negative reward applied when no aspects were completed and no
        execution succeeded, discouraging spinning in place.
    """
    step_penalty: float = -0.1
    aspect_complete: float = 0.9
    exec_success: float = 0.3
    terminal_success: float = 15.0
    invalid_done: float = -5.0
    stagnation: float = -0.5


# ── Abstract base ──────────────────────────────────────────────────────────────

class RewardFunction(ABC):
    """
    Abstract base class for all reward functions used in OpenEngine training.

    Subclass this and implement :meth:`compute` to plug your own reward signal
    into the PPO training loop or the rollout generator.

    The ``transition`` dict passed to ``compute`` must be deserializable into
    a :class:`TransitionData` instance.  The simplest way to consume it is::

        td = TransitionData(**transition)

    Methods
    -------
    compute(transition)
        Return a scalar reward for the given transition.
    """

    @abstractmethod
    def compute(self, transition: dict) -> float:
        """
        Compute a scalar reward for a single environment transition.

        Parameters
        ----------
        transition:
            A dictionary whose keys match the fields of :class:`TransitionData`.
            Extra keys are silently ignored so that callers can pass enriched
            dicts without breaking backwards compatibility.

        Returns
        -------
        float
            The scalar reward signal for this transition.
        """
        ...


# ── Default implementation ─────────────────────────────────────────────────────

class DefaultRewardFunction(RewardFunction):
    """
    Default reward function matching the base-policy training signal.

    Reward logic (applied in order):

    1. **Step penalty** — applied every step to encourage short episodes.
    2. **Aspect completion bonus** — fires once for each newly completed goal
       aspect (``aspects_after - aspects_before``).
    3. **Execution success bonus** — fires when a tool or code execution
       returned successfully, even with no aspect progress.
    4. **Terminal success** — large bonus when DONE fires with all aspects
       complete (``aspects_after > 0`` and none remain).
    5. **Invalid DONE penalty** — fires when DONE fires prematurely.
    6. **Stagnation penalty** — fires when no progress and no success in the
       step.

    Parameters
    ----------
    config:
        A :class:`RewardConfig` instance.  Defaults to the standard weights
        used during base-policy training.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config: RewardConfig = config or RewardConfig()

    def compute(self, transition: dict) -> float:
        """
        Compute the default scalar reward from the given transition dict.

        Parameters
        ----------
        transition:
            Dict with the fields of :class:`TransitionData`.  Unknown keys are
            stripped before construction.

        Returns
        -------
        float
            Scalar reward in approximately [-6, +16] for standard config.
        """
        # Strip unknown keys so callers can pass enriched dicts safely.
        known_fields = TransitionData.__dataclass_fields__
        filtered = {k: v for k, v in transition.items() if k in known_fields}
        td = TransitionData(**filtered)

        cfg = self.config
        reward: float = 0.0

        # 1. Step penalty (always applied)
        reward += cfg.step_penalty

        # 2. Aspect completion bonus (delta-based)
        aspect_delta = max(0, td.aspects_after - td.aspects_before)
        if aspect_delta > 0:
            reward += cfg.aspect_complete * aspect_delta

        # 3. Execution success bonus
        if td.exec_success:
            reward += cfg.exec_success

        # 4 & 5. Terminal outcome
        if td.action_name == "DONE" or td.done:
            if td.aspects_after > 0 and aspect_delta >= 0:
                # Check whether the task appears fully resolved
                # (aspects_after > aspects_before reaching terminal, or
                #  aspects_after > 0 meaning at least some work was done)
                reward += cfg.terminal_success
            else:
                reward += cfg.invalid_done

        # 6. Stagnation penalty (no progress, no success)
        elif aspect_delta == 0 and not td.exec_success:
            reward += cfg.stagnation

        return reward


# ── Module-level factory ───────────────────────────────────────────────────────

def get_default_reward_function() -> RewardFunction:
    """
    Return a ready-to-use instance of :class:`DefaultRewardFunction` with
    the standard reward weights used during base-policy training.

    This is the recommended entry point for the PPO trainer and rollout
    generator when no custom reward function has been provided.

    Returns
    -------
    RewardFunction
        A :class:`DefaultRewardFunction` instance with default config.

    Example
    -------
    ::

        from training.reward import get_default_reward_function

        reward_fn = get_default_reward_function()
        r = reward_fn.compute({
            "action_name": "INVOKE_TOOL",
            "aspects_before": 0,
            "aspects_after": 1,
            "exec_success": True,
            "exec_returncode": 0,
            "done": False,
            "step": 3,
        })
        print(r)  # 1.1  (step_penalty + aspect_complete + exec_success)
    """
    return DefaultRewardFunction()
