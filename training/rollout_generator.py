import sys
from pathlib import Path

# ── PATH BOOTSTRAP (MUST BE FIRST) ────────────────────────────────────────────
# Add the Reinforcement/ directory so xyber_env and rl_chassis are importable
# regardless of whether this file is run directly or as python -m Reinforcement.rollout_generator.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import os
import glob
import random
import logging
from typing import Dict, Any
import numpy as np
import torch
from xyber_env import XybernetexEnv
from rl_chassis import ActionSpace, _is_local_goal, _is_code_goal, _is_tool_goal, _is_multi_tool_goal

# Configure logging
logging.basicConfig(level=logging.ERROR)

# ── SYNTHETIC TRAINING GOALS ──────────────────────────────────────────────────
# Five categories:
#   CODE goals          — PLAN → WRITE(CODE) → EXEC → DONE
#   STRATEGIC goals     — PLAN → WRITE(DOC)  → DONE
#   WEB RESEARCH goals  — PLAN → INVOKE_TOOL(search) → WRITE(DOC) → DONE
#   EMAIL goals         — PLAN → INVOKE_TOOL(email)  → WRITE(DOC) → DONE
#   RESEARCH+EMAIL goals— PLAN → INVOKE_TOOL(search) → ANALYZE
#                              → INVOKE_TOOL(email)  → WRITE(DOC) → DONE

CODE_GOALS = [
    "Create a directory named 'XyberTest' and write a status.txt file inside it.",
    "Write a script to calculate the Fibonacci sequence up to 100.",
    "Create a shell script that lists all environment variables.",
    "Write a Python class that manages a simple inventory system.",
    "Build a Python function that sorts a list of dictionaries by a specified key.",
    "Write a Python script that counts word frequency in a given text file.",
    "Create a directory structure for a new microservice project.",
    "Build a Python class that implements a basic stack data structure.",
    "Write an executable script that validates JSON files in a directory.",
    "Build a command-line tool that converts CSV to JSON format.",
]

STRATEGIC_GOALS = [
    "Analyze the pros and cons of microservices versus monolithic architecture.",
    "Compare PostgreSQL and MongoDB for a real-time analytics workload.",
    "Evaluate the risk factors for deploying an AI agent in a production environment.",
    "Summarize the key trade-offs between serverless and container-based deployment.",
    "Assess the competitive landscape for cloud-native database providers.",
    "Analyze the implications of the EU AI Act on autonomous agent deployment.",
    "Compare event-driven and request-driven architectures for a high-throughput API.",
    "Evaluate whether a startup should adopt a multi-cloud strategy in year one.",
    "Assess the long-term maintenance costs of a custom ML pipeline vs managed services.",
    "Analyze the trade-offs between consistency and availability in distributed systems.",
]

WEB_GOALS = [
    "Search the web for the latest Python 3.12 release notes.",
    "Research current best practices for PostgreSQL connection pooling in production.",
    "Find and summarize the top 3 trending open-source AI frameworks this month.",
    "Look up the current market share of major cloud providers and compare pricing.",
    "Search for recent security advisories affecting Node.js applications.",
    "Research the differences between REST and GraphQL for API design decisions.",
    "Find the latest benchmarks comparing PyTorch and TensorFlow inference speed.",
    "Look up OWASP top 10 vulnerabilities and summarize mitigation strategies.",
    "Research best practices for deploying containers on OVH managed Kubernetes.",
    "Search for recent advances in reinforcement learning from human feedback.",
]

EMAIL_GOALS = [
    "Send an email to the team notifying them that the deployment is complete.",
    "Email a summary of this week's engineering progress to chris@xybernetex.com.",
    "Notify the client via email that their report is ready for review.",
    "Send a status update email to the stakeholders about the current sprint.",
    "Email the on-call engineer to notify them of the scheduled maintenance window.",
    "Send a welcome email to the new team member joining the platform team.",
    "Notify chris@xybernetex.com via email that the data export has finished.",
    "Email a brief incident report to the operations team.",
    "Send a reminder email to all users about the upcoming system upgrade.",
    "Notify the sales team via email that a new lead has been added to the CRM.",
]

RESEARCH_EMAIL_GOALS = [
    "Research the best countries to relocate manufacturing to as an alternative to China given current US tariffs and email the findings to chris@xybernetex.com.",
    "Search for the latest AI agent frameworks released in 2025 and email a comparison report to chris@xybernetex.com.",
    "Research current PostgreSQL best practices for high-availability deployments and email a summary to the engineering team.",
    "Find the top 5 open-source alternatives to Notion and email a structured comparison to chris@xybernetex.com.",
    "Look up recent GDPR enforcement actions in 2025 and email a compliance briefing to the legal team.",
    "Research the current state of quantum computing hardware and email a strategic briefing to chris@xybernetex.com.",
    "Find the latest benchmarks for LLM inference on consumer GPUs and email the results to the ML team.",
    "Research alternatives to AWS S3 for object storage and send a cost comparison report to chris@xybernetex.com.",
    "Look up the current venture capital landscape for AI infrastructure startups and email a market brief to chris@xybernetex.com.",
    "Search for the best practices in zero-downtime database migrations and email a playbook to the engineering team.",
]

TRAINING_GOALS = CODE_GOALS + STRATEGIC_GOALS + WEB_GOALS + EMAIL_GOALS + (RESEARCH_EMAIL_GOALS * 8)


class SmarterTeacher:
    """
    Task-aware heuristic expert demonstrating the Golden Path.

    Golden paths by goal type:
      CODE         — PLAN → WRITE_ARTIFACT(CODE) → EXECUTE_CODE → DONE
      TOOL         — PLAN → INVOKE_TOOL → WRITE_ARTIFACT(DOC) → DONE
      MULTI-TOOL   — PLAN → INVOKE_TOOL(search) → ANALYZE
                          → INVOKE_TOOL(email)  → WRITE_ARTIFACT(DOC) → DONE
      OTHER        — PLAN → WRITE_ARTIFACT(DOC) → DONE

    Inhibition rules (no stagnation):
      - Repeating WRITE_ARTIFACT  → fall back to ANALYZE
      - Repeating EXECUTE_CODE    → fall back to ANALYZE
      - Repeating PLAN/ANALYZE    → fall back to WRITE_ARTIFACT if aspects
                                    are pending, otherwise ANALYZE
      - Local goals               → INVOKE_TOOL fully suppressed (hard zero)

    Multi-tool behaviour:
      INVOKE_TOOL fires up to _max_invocations times per episode.
      Between consecutive invocations, at least one non-tool step is required
      (ANALYZE) so the policy learns to process results before re-dispatching.
    """

    def __init__(self, goal: str = ""):
        self.prev_action        = None
        self._local             = _is_local_goal(goal)
        self._code              = _is_code_goal(goal)
        self._tool              = _is_tool_goal(goal)
        self._multi             = _is_multi_tool_goal(goal)
        self._invoke_count      = 0   # how many times INVOKE_TOOL has fired
        self._max_invocations   = 2 if self._multi else 1
        self._last_was_invoke   = False  # enforce gap between invocations

    def set_goal(self, goal: str) -> None:
        """Call once per episode reset so the teacher re-evaluates task type."""
        self._local           = _is_local_goal(goal)
        self._code            = _is_code_goal(goal)
        self._tool            = _is_tool_goal(goal)
        self._multi           = _is_multi_tool_goal(goal)
        self._invoke_count    = 0
        self._max_invocations = 2 if self._multi else 1
        self._last_was_invoke = False
        self.prev_action      = None

    def _can_invoke(self) -> bool:
        """True if INVOKE_TOOL budget remains and the last step was not also a tool call."""
        return (
            self._invoke_count < self._max_invocations
            and not self._last_was_invoke
        )

    def select_action(self, info: Dict[str, Any]) -> int:
        has_pending       = info.get("has_pending_aspects", False)
        delivery_pending  = info.get("delivery_pending",   False)

        # ── Terminal check ────────────────────────────────────────────────────
        if info.get("all_aspects_complete", False):
            return ActionSpace.encode("DONE")

        # ── Primary action selection ───────────────────────────────────────────

        # Delivery pending: the ONLY remaining aspects require a tool call.
        # Fire INVOKE_TOOL regardless of invocation count gap rules —
        # this is the correct and only valid action in this state.
        if delivery_pending and not self._local and self._can_invoke():
            action = ActionSpace.encode("INVOKE_TOOL")

        elif self._code and info.get("has_unexecuted_code", False):
            action = ActionSpace.encode("EXECUTE_CODE")

        elif self._tool and has_pending and self._can_invoke() and self._invoke_count == 0:
            # First tool invocation (web search / data fetch).
            # Only fires on the first call — subsequent tool calls are gated
            # by delivery_pending above, preventing premature second invocations.
            action = ActionSpace.encode("INVOKE_TOOL")

        elif has_pending:
            action = ActionSpace.encode("WRITE_ARTIFACT")

        elif info.get("needs_planning", True):
            action = ActionSpace.encode("PLAN")

        else:
            action = ActionSpace.encode("ANALYZE")

        # ── Track INVOKE_TOOL budget ───────────────────────────────────────────
        if action == ActionSpace.encode("INVOKE_TOOL"):
            self._invoke_count    += 1
            self._last_was_invoke  = True
        else:
            self._last_was_invoke  = False

        # ── Action inhibition (anti-stagnation) ───────────────────────────────
        if action == self.prev_action:
            write  = ActionSpace.encode("WRITE_ARTIFACT")
            exec_  = ActionSpace.encode("EXECUTE_CODE")
            plan   = ActionSpace.encode("PLAN")
            anl    = ActionSpace.encode("ANALYZE")
            invoke = ActionSpace.encode("INVOKE_TOOL")

            if action == write:
                action = anl
            elif action == exec_:
                action = anl
            elif action in (plan, anl):
                # Break stagnation: tool goals use INVOKE_TOOL if budget remains
                if not self._local and self._can_invoke():
                    action = invoke
                else:
                    action = write if has_pending else anl

        # ── Hard mathematical zero: local goals NEVER emit INVOKE_TOOL ───────
        # The expert's probability of selecting INVOKE_TOOL for a local goal is
        # exactly 0.  The policy must learn: s[63]=1.0 ⇒ P(INVOKE_TOOL) = 0.
        if self._local and action == ActionSpace.encode("INVOKE_TOOL"):
            action = ActionSpace.encode("WRITE_ARTIFACT") if has_pending \
                     else ActionSpace.encode("ANALYZE")

        self.prev_action = action
        return action

def _purge_rollout_data(data_dir: str) -> int:
    """Delete all .npz files in data_dir. Returns count removed."""
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    for f in files:
        os.remove(f)
    return len(files)


def run_expert_generation(num_episodes: int = 1500, output_file: str = "data/expert_rollouts.npz"):
    # Purge stale demonstrations before generating corrected ones
    data_dir = os.path.dirname(output_file)
    os.makedirs(data_dir, exist_ok=True)
    purged = _purge_rollout_data(data_dir)
    if purged:
        print(f"[Purge] Deleted {purged} stale rollout file(s) from {data_dir}/")

    env = XybernetexEnv(mock_llm=True)
    teacher = SmarterTeacher()

    states, actions, rewards, next_states, dones = [], [], [], [], []

    print(f"--- Xybernetex Expert Rollout Initialization ---")
    print(f"[System] Target: {num_episodes} episodes.")
    print(f"[System] Goal mix: {len(CODE_GOALS)} code | {len(STRATEGIC_GOALS)} strategic | "
          f"{len(WEB_GOALS)} web | {len(EMAIL_GOALS)} email | {len(RESEARCH_EMAIL_GOALS)} research+email")

    total_steps = 0
    for ep in range(num_episodes):
        current_goal = random.choice(TRAINING_GOALS)
        # Notify teacher of new goal so it can re-evaluate task type and
        # suppress INVOKE_TOOL for local-execution goals.
        teacher.set_goal(current_goal)
        state = env.reset(goal=current_goal)
        done = False
        
        while not done:
            info_for_teacher = env.get_expert_info()
            action = teacher.select_action(info_for_teacher)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            
            state = next_state
            total_steps += 1

        if (ep + 1) % 25 == 0:
            print(f"  Progress: {ep + 1}/{num_episodes} episodes complete.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savez_compressed(
        output_file,
        states=np.array(states, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        next_states=np.array(next_states, dtype=np.float32),
        dones=np.array(dones, dtype=bool)
    )

    action_names = ["PLAN", "ANALYZE", "INVOKE_TOOL", "EXECUTE_CODE", "WRITE_ARTIFACT", "DONE"]
    action_counts = np.bincount(np.array(actions, dtype=np.int64), minlength=6)

    print(f"\n[Success] Dataset generated: {output_file}")
    print(f"  - Total Transitions:       {len(states)}")
    print(f"  - Mean Steps Per Episode:  {len(states) / num_episodes:.2f}")
    print(f"  - Action Distribution:")
    for i, name in enumerate(action_names):
        pct = 100.0 * action_counts[i] / len(states)
        print(f"      {name:<16} {action_counts[i]:>5}  ({pct:.1f}%)")

    env.close()

if __name__ == "__main__":
    run_expert_generation()