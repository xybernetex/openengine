"""
rl_chassis.py — Xybernetex Reinforcement Learning Chassis

Mathematical bridge between WorkingMemory/WorldEngine state and the PPO
training loop.  Zero network calls, zero I/O side effects, zero terminal UI.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ── Intent detection vocabulary ───────────────────────────────────────────────
# Canonical source — rollout_generator.py and xyber_env.py import from here.
_LOCAL_KEYWORDS: frozenset[str] = frozenset({
    "directory", "folder", "file", "write", "local", "script",
    "create", "generate", "calculate", "fibonacci", "class", "function",
    "executable", "run", "execute", "shell", "bash", "powershell",
    "build", "sort", "parse", "inventory", "list", "environment",
})

# Code-specific keywords — goal requires CODE artifact + EXECUTE_CODE to complete.
# Strategic/analysis goals (analyze, compare, evaluate) do NOT require execution.
_CODE_KEYWORDS: frozenset[str] = frozenset({
    "script", "code", "function", "class", "calculate", "fibonacci",
    "sort", "parse", "implement", "program", "algorithm",
    "execute", "run", "automate", "compute", "build",
    "directory", "file", "folder", "write",
})


def _is_local_goal(goal: str) -> bool:
    """Return True if the goal is purely local/generative (no web needed)."""
    return bool(_LOCAL_KEYWORDS.intersection(goal.lower().split()))


def _is_code_goal(goal: str) -> bool:
    """Return True if the goal requires code execution to complete."""
    return bool(_CODE_KEYWORDS.intersection(goal.lower().split()))


# Tool-specific keywords — goal requires an external tool call (web search,
# webhook, API, CRM, notification, etc.).  Must NOT overlap with local goals.
_TOOL_KEYWORDS: frozenset[str] = frozenset({
    "search", "research", "find", "look", "web", "online",
    "latest", "current", "recent", "news", "trends", "trending",
    "browse", "fetch", "retrieve", "lookup",
    "send", "notify", "post", "email", "slack", "crm",
    "integration", "webhook", "api",
})


def _is_tool_goal(goal: str) -> bool:
    """Return True if the goal requires an external tool call.

    A tool goal is one that matches _TOOL_KEYWORDS and is NOT a local/code goal —
    the two intent classes are mutually exclusive by design.
    """
    words = goal.lower().split()
    return (
        bool(_TOOL_KEYWORDS.intersection(words))
        and not _is_local_goal(goal)
    )


_SEARCH_KEYWORDS: frozenset[str] = frozenset({
    "search", "research", "find", "look", "web", "online",
    "latest", "current", "recent", "news", "trends", "trending",
    "browse", "fetch", "retrieve", "lookup",
})

_EMAIL_KEYWORDS: frozenset[str] = frozenset({
    "email", "send", "notify", "mail", "message", "report",
})

# Delivery-aspect keywords — aspects whose names contain these words can ONLY
# be completed by a successful INVOKE_TOOL call (e.g. send_email).
# WRITE_ARTIFACT must skip them; the policy must learn to re-invoke the tool.
# Exact canonical names for delivery aspects — keyword scanning is intentionally
# avoided here.  The PLAN LLM sometimes produces aspects like 'draft_email_summary'
# that contain delivery keywords but are pure content tasks.  Treating those as
# delivery-locked causes multiple INVOKE_TOOL calls (and multiple emails) per run.
_DELIVERY_ASPECT_NAMES: frozenset[str] = frozenset({
    "email_transmission_confirmation",
    "sms_delivery_confirmation",
    "push_notification_confirmation",
    "webhook_delivery_confirmation",
})


def _is_delivery_aspect(aspect_name: str) -> bool:
    """Return True if this aspect requires INVOKE_TOOL (not WRITE_ARTIFACT) to complete."""
    return aspect_name.lower().strip() in _DELIVERY_ASPECT_NAMES


def _is_search_goal(goal: str) -> bool:
    """Return True if the goal requires a web search, regardless of whether it
    is also a code/local goal.

    Unlike _is_tool_goal, this does NOT exclude local/code goals — a hybrid goal
    like "research X, write a Python script, and email the results" needs
    INVOKE_TOOL for the search phase even though it also needs EXECUTE_CODE.
    """
    words = set(goal.lower().split())
    return bool(_SEARCH_KEYWORDS.intersection(words))


def _is_multi_tool_goal(goal: str) -> bool:
    """Return True if the goal requires multiple sequential tool invocations.

    Canonical example: research a topic AND email the results.
    The policy must learn to emit INVOKE_TOOL twice — once for the search,
    once for the delivery — with at least one intermediate step between them.
    """
    words = set(goal.lower().split())
    has_search = bool(_SEARCH_KEYWORDS.intersection(words))
    has_email  = bool(_EMAIL_KEYWORDS.intersection(words))
    return has_search and has_email and not _is_local_goal(goal)


# ── Dimensionality constants ───────────────────────────────────────────────────

STATE_DIM    : int = 256
ASPECT_CAP   : int = 10          # mirrors working_memory.ASPECT_CAP
ACTION_COUNT : int = 6
ACTION_DIM   : int = ACTION_COUNT  # alias used by ppo_trainer / inference_worker
HISTORY_LEN  : int = 32          # steps of one-hot action history kept in state
_STRUCT_DIM  : int = 64          # structural block  (STATE_DIM − HISTORY_LEN × ACTION_COUNT)

assert HISTORY_LEN * ACTION_COUNT + _STRUCT_DIM == STATE_DIM, "Dimension mismatch"


# ── Neural network ─────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Shared trunk → actor + critic heads.
    Architecture is fixed; changing it invalidates saved weights.

    Trunk:  256 → 256 (GELU + LayerNorm) → 128 (GELU + LayerNorm)
    Actor:  128 → action_dim   (logits, no softmax)
    Critic: 128 → 1            (state value estimate)
    """

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        self.actor  = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        features = self.trunk(x)
        return self.actor(features), self.critic(features)


# ══════════════════════════════════════════════════════════════════════════════
# ACTION SPACE
# ══════════════════════════════════════════════════════════════════════════════

class ActionSpace:
    """
    Discrete action set for the Xybernetex MDP.
    Index → primitive mapping is fixed; changing it invalidates saved weights.
    """

    PRIMITIVES: tuple[str, ...] = (
        "PLAN",           # 0 — decompose goal into trackable aspects
        "ANALYZE",        # 1 — produce a structured ANALYSIS artifact
        "INVOKE_TOOL",    # 2 — call any registered webhook tool (web search, CRM, etc.)
        "EXECUTE_CODE",   # 3 — run the most recent CODE artifact in sandbox
        "WRITE_ARTIFACT", # 4 — produce a CODE / DOCUMENT / NARRATIVE artifact
        "DONE",           # 5 — assert task completion (validated by env)
    )

    N: int = len(PRIMITIVES)
    _IDX: dict[str, int] = {p: i for i, p in enumerate(PRIMITIVES)}

    @classmethod
    def decode(cls, index: int) -> str:
        """int → action string."""
        if not 0 <= index < cls.N:
            raise ValueError(f"Invalid action index {index}")
        return cls.PRIMITIVES[index]

    @classmethod
    def encode(cls, action: str) -> int:
        """action string → int."""
        try:
            return cls._IDX[action]
        except KeyError:
            raise ValueError(f"Unknown action '{action}'")

    @classmethod
    def sample(cls) -> int:
        """Uniform random action."""
        return int(np.random.randint(0, cls.N))

    @classmethod
    def mask(cls, valid: list[str]) -> np.ndarray:
        """Boolean mask over the action space. True = action available."""
        m = np.zeros(cls.N, dtype=bool)
        for a in valid:
            if a in cls._IDX:
                m[cls._IDX[a]] = True
        return m


# ══════════════════════════════════════════════════════════════════════════════
# STATE VECTORIZER
# ══════════════════════════════════════════════════════════════════════════════

class StateVectorizer:
    """
    Converts WorkingMemory state into a deterministic 256-dim float32 vector.
    """

    _MAX_ARTIFACTS : float = 30.0
    _MAX_ERRORS    : float = 20.0
    _MAX_NODES     : float = 80.0
    _MAX_EDGES     : float = 150.0
    _MAX_COVERS    : float = 30.0
    _MAX_AEU       : float = 50.0
    _MAX_SAFETY    : float = 10.0
    _MAX_DONE_FAIL : float = 5.0

    _ARTIFACT_TYPES: tuple[str, ...] = (
        "CODE", "DOCUMENT", "NARRATIVE",
        "ANALYSIS", "DIAGRAM", "DECISION", "REPLAN",
    )
    _STATUS_FLOAT: dict[str, float] = {
        "pending":     0.1,
        "in_progress": 0.5,
        "complete":    1.0,
    }

    def __init__(self) -> None:
        self._history: np.ndarray = np.zeros((HISTORY_LEN, ACTION_COUNT), dtype=np.float32)

    def reset(self) -> None:
        self._history[:] = 0.0

    def push_action(self, action_index: int) -> None:
        self._history = np.roll(self._history, shift=-1, axis=0)
        self._history[-1, :] = 0.0
        self._history[-1, action_index] = 1.0

    def vectorize(self, conn: sqlite3.Connection, run_id: str, step_number: int, max_steps: int, goal: str = "") -> np.ndarray:
        structural = self._build_structural(conn, run_id, step_number, max_steps, goal)
        history    = self._history.flatten()
        vec = np.concatenate([structural, history]).astype(np.float32)
        return np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=0.0)

    def _build_structural(self, conn: sqlite3.Connection, run_id: str, step_number: int, max_steps: int, goal: str = "") -> np.ndarray:
        s = np.zeros(_STRUCT_DIM, dtype=np.float32)

        # [0:10] Aspect completion (10 slots for up to ASPECT_CAP aspects)
        aspects = self._q(conn, "SELECT aspect, status, blocking_error_id FROM goal_aspects WHERE run_id=? ORDER BY created_at", (run_id,))
        for k in range(ASPECT_CAP):
            if k < len(aspects):
                s[k] = self._STATUS_FLOAT.get(aspects[k]["status"], 0.0)

        # [10:15] EXPERT-MIRROR SIGNALS — the exact booleans the SmarterTeacher
        # uses to decide actions.  These make the decision boundary trivially
        # learnable: the network no longer has to reverse-engineer them from
        # indirect artifact counts and individual aspect floats.
        status_set = {a["status"] for a in aspects}
        n_pending     = sum(1 for a in aspects if a["status"] == "pending")
        n_in_progress = sum(1 for a in aspects if a["status"] == "in_progress")
        n_complete    = sum(1 for a in aspects if a["status"] == "complete")

        s[10] = 1.0 if len(aspects) == 0 else 0.0                         # needs_planning
        s[11] = 1.0 if n_pending > 0 else 0.0                             # has_pending_aspects
        s[12] = 1.0 if len(aspects) > 0 and n_complete == len(aspects) else 0.0  # all_aspects_complete
        s[13] = 1.0 if n_in_progress > 0 else 0.0                         # has_in_progress_aspects

        # [14] has_unexecuted_code — the CRITICAL missing signal
        unexecuted = self._q(conn,
            "SELECT COUNT(*) AS n FROM artifacts a "
            "LEFT JOIN executions e ON a.id = e.artifact_id "
            "WHERE a.run_id = ? AND a.artifact_type = 'CODE' AND e.id IS NULL",
            (run_id,))
        s[14] = 1.0 if (unexecuted and unexecuted[0]["n"] > 0) else 0.0   # has_unexecuted_code

        # [15] mean aspect completion
        if aspects:
            s[15] = float(np.mean([self._STATUS_FLOAT.get(a["status"], 0.0) for a in aspects]))

        # [16] code_goal bit — tells the policy whether EXECUTE_CODE is needed
        # 1.0 = goal requires code execution (WRITE→EXEC chain)
        # 0.0 = strategic/analysis goal (WRITE alone completes aspects)
        s[16] = 1.0 if _is_code_goal(goal) else 0.0

        # [17] tool_goal bit — tells the policy whether INVOKE_TOOL is needed
        # 1.0 = goal requires external tool (web search, webhook, API, CRM, etc.)
        # 0.0 = local/strategic goal (no external tool call required)
        # Mirrors s[16] pattern; hard-masked at inference when 0.0.
        s[17] = 1.0 if _is_tool_goal(goal) else 0.0

        # [18] delivery_pending — every remaining pending aspect requires INVOKE_TOOL.
        # When True the policy must select INVOKE_TOOL; WRITE_ARTIFACT cannot advance state.
        pending_names = [a["aspect"] for a in aspects if a["status"] == "pending"]
        s[18] = 1.0 if (pending_names and all(_is_delivery_aspect(n) for n in pending_names)) else 0.0

        # [19] search_goal bit — 1.0 if the goal requires a web search.
        # Unlike s[17] (tool_goal), this is NOT mutually exclusive with code goals.
        # A hybrid goal (research + python + email) sets both s[16] AND s[19].
        # Hard-masked at inference when 0.0 to prevent spurious web_search calls
        # on pure code/analysis goals.
        s[19] = 1.0 if _is_search_goal(goal) else 0.0

        # [20:30] Blocked flags (10 slots for up to ASPECT_CAP aspects)
        for k in range(ASPECT_CAP):
            if k < len(aspects):
                s[20 + k] = 1.0 if aspects[k]["blocking_error_id"] else 0.0

        # [30] search_done bit — 1.0 if at least one NARRATIVE artifact already
        # exists for this run (i.e. a web_search INVOKE_TOOL has already fired).
        # Used by the inference mask to block additional web searches after the
        # first one completes, forcing the policy to move on to WRITE_ARTIFACT.
        narrative_count = self._q(conn,
            "SELECT COUNT(*) AS n FROM artifacts "
            "WHERE run_id = ? AND artifact_type = 'NARRATIVE'",
            (run_id,))
        s[30] = 1.0 if (narrative_count and narrative_count[0]["n"] > 0) else 0.0

        # [31:40] reserved for future structural features

        # [40:42] Step budget
        horizon = max(max_steps, 1)
        s[40] = step_number / horizon
        s[41] = (max_steps - step_number) / horizon

        # [42:45] Error telemetry
        errors = self._q(conn, "SELECT resolved FROM errors WHERE run_id=?", (run_id,))
        total_err = len(errors)
        resolved = sum(1 for e in errors if e["resolved"])
        s[42] = min(total_err / self._MAX_ERRORS, 1.0)
        s[43] = min((total_err - resolved) / self._MAX_ERRORS, 1.0)
        s[44] = resolved / max(total_err, 1)

        # [45:48] Execution stats
        execs = self._q(conn, "SELECT e.success, e.timed_out, e.returncode FROM executions e JOIN artifacts a ON a.id = e.artifact_id WHERE a.run_id=? ORDER BY e.executed_at DESC LIMIT 8", (run_id,))
        if execs:
            s[45] = sum(1 for e in execs if e["success"]) / len(execs)
            s[46] = sum(1 for e in execs if e["timed_out"]) / len(execs)
            s[47] = 1.0 if any(e["returncode"] == 0 for e in execs) else 0.0

        # [48:55] Type distribution
        arts = self._q(conn, "SELECT artifact_type, is_duplicate FROM artifacts WHERE run_id=?", (run_id,))
        type_counts = {t: 0 for t in self._ARTIFACT_TYPES}
        dup_count = 0
        for art in arts:
            if art["artifact_type"] in type_counts: type_counts[art["artifact_type"]] += 1
            if art["is_duplicate"]: dup_count += 1
        for k, t in enumerate(self._ARTIFACT_TYPES):
            s[48 + k] = min(type_counts[t] / self._MAX_ARTIFACTS, 1.0)

        # [55:57] Aggregate
        s[55] = min(len(arts) / self._MAX_ARTIFACTS, 1.0)
        s[56] = dup_count / max(len(arts), 1)

        # [57:60] Graph metrics
        node_rows = self._q(conn, "SELECT COUNT(DISTINCT source_id) + COUNT(DISTINCT target_id) AS n FROM artifact_edges WHERE run_id=?", (run_id,))
        edges = self._q(conn, "SELECT rel_type FROM artifact_edges WHERE run_id=?", (run_id,))
        s[57] = min((node_rows[0]["n"] if node_rows else 0) / self._MAX_NODES, 1.0)
        s[58] = min(len(edges) / self._MAX_EDGES, 1.0)
        s[59] = min(sum(1 for e in edges if e["rel_type"] == "COVERS") / self._MAX_COVERS, 1.0)

        # [60] AEU [61] Safety [62] Done Fail [63] Intent Bit (local=1.0, web=0.0)
        aeu = self._q(conn, "SELECT COUNT(DISTINCT content_hash) AS n FROM aeu WHERE run_id=?", (run_id,))
        s[60] = min((aeu[0]["n"] if aeu else 0) / self._MAX_AEU, 1.0)
        safety = self._q(conn, "SELECT COUNT(*) AS n FROM safety_events WHERE run_id=?", (run_id,))
        s[61] = min((safety[0]["n"] if safety else 0) / self._MAX_SAFETY, 1.0)
        done_fail = self._q(conn, "SELECT COUNT(*) AS n FROM termination_events WHERE run_id=? AND allowed=0", (run_id,))
        s[62] = min((done_fail[0]["n"] if done_fail else 0) / self._MAX_DONE_FAIL, 1.0)

        # Intent bit — the neural network sees 1.0 at s[63] for local-execution
        # goals and 0.0 for web-research goals from step 0 onward.
        s[63] = 1.0 if _is_local_goal(goal) else 0.0

        return s

    @staticmethod
    def _q(conn: sqlite3.Connection, sql: str, params: tuple) -> list[dict]:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


# ══════════════════════════════════════════════════════════════════════════════
# REWARD FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TransitionInfo:
    action_index      : int
    prev_aspect_states: dict[str, str]
    curr_aspect_states: dict[str, str]
    exec_success      : bool  = False
    exec_timed_out    : bool  = False
    exec_returncode   : int   = -1
    artifact_written  : bool  = False
    artifact_is_duplicate: bool = False
    aeu_gain          : int   = 0
    safety_triggered  : bool  = False
    done_valid        : bool  = False
    done_blockers     : list[str] = field(default_factory=list)

class RewardFunction:
    """
    High-Stakes Reward Shaper. 
    Calibrated to crush loops and enforce surgical efficiency.
    """
    R_TERMINAL_VALID   : float =  15.0   # Dominant success signal
    R_ASPECT_COMPLETE  : float =   1.0   # Progress signal
    R_EXEC_SUCCESS     : float =   0.5   # Functional verification
    R_AEU_GAIN         : float =   0.2   # Info gain per new AEU
    
    # Penalties
    R_STEP_PENALTY     : float =  -0.1   # Living cost
    R_STAGNATION       : float =  -0.75  # 7.5x increase to kill repeat loops
    R_DUPLICATE        : float =  -0.5   # Punish redundant artifacts
    R_EXEC_FAIL        : float =  -0.5   # Code errors
    R_SAFETY           : float =  -1.0   # Safety violations
    R_DONE_INVALID     : float =  -5.0   # Harsh penalty for guessing DONE

    def compute(self, info: TransitionInfo, prev_action_index: int | None) -> float:
        r: float = self.R_STEP_PENALTY

        # 1. Termination (Exit immediately if DONE)
        if info.action_index == ActionSpace.encode("DONE"):
            return self.R_TERMINAL_VALID if info.done_valid else self.R_DONE_INVALID

        # 2. Aspect Transitions
        for aspect, curr_status in info.curr_aspect_states.items():
            prev_status = info.prev_aspect_states.get(aspect, "pending")
            if curr_status == "complete" and prev_status != "complete":
                r += self.R_ASPECT_COMPLETE

        # 3. Execution & Quality
        if info.action_index == ActionSpace.encode("EXECUTE_CODE"):
            if info.exec_success and info.exec_returncode == 0:
                r += self.R_EXEC_SUCCESS
            elif not info.exec_success or info.exec_timed_out:
                r += self.R_EXEC_FAIL

        r += (self.R_AEU_GAIN * info.aeu_gain)
        if info.artifact_is_duplicate: r += self.R_DUPLICATE
        if info.safety_triggered: r += self.R_SAFETY

        # 4. Stagnation (The Loop Killer)
        if prev_action_index is not None and info.action_index == prev_action_index:
            r += self.R_STAGNATION

        # 5. Delivery-lock penalty
        # All remaining pending aspects are tool-locked (delivery_pending=True)
        # but the agent chose something other than INVOKE_TOOL.
        # WRITE_ARTIFACT cannot advance state; ANALYZE burns a step.
        # Apply a strong penalty so the policy learns s[18]=1.0 → INVOKE_TOOL.
        delivery_pending = (
            bool(info.prev_aspect_states)
            and all(
                _is_delivery_aspect(k)
                for k, v in info.prev_aspect_states.items()
                if v == "pending"
            )
            and any(v == "pending" for v in info.prev_aspect_states.values())
        )
        invoke_tool_idx = ActionSpace.encode("INVOKE_TOOL")
        if delivery_pending and info.action_index != invoke_tool_idx:
            r += self.R_STAGNATION   # -0.75 on top of step penalty = -0.85 per wasted step

        return r