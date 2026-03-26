"""
sdk/engine.py — Xybernetex OpenEngine public SDK.

The primary entry point for using OpenEngine programmatically.

Usage
-----
    from sdk.engine import Engine

    engine = Engine()  # reads LLM credentials from .env

    result = engine.run(
        goal="Research the top 5 EV battery suppliers and email a report to me@company.com",
    )

    print(result.status)           # "completed" | "failed" | "cancelled"
    print(result.success)          # True / False
    print(result.steps)            # 6
    print(result.reward)           # 21.0

    for artifact in result.artifacts:
        print(artifact.type, artifact.title)
        print(artifact.content[:200])

    code = result.first("CODE")    # first CODE artifact, or None
    if code:
        exec(compile(code.content, "<agent>", "exec"))

Workflow mode (skip PLAN, user defines structure)
-------------------------------------------------
    result = engine.run(
        goal="Research lithium supply chain risks and email findings to me@company.com",
        aspects=["lithium_supply_chain_research", "email_transmission_confirmation"],
    )
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure the repo root (parent of sdk/) is on sys.path so engine imports resolve.
_SDK_DIR  = Path(__file__).resolve().parent
_REPO_ROOT = _SDK_DIR.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class Artifact:
    """A single artifact produced during a goal run."""
    type: str          # "CODE" | "DOCUMENT" | "NARRATIVE" | "ANALYSIS"
    title: str
    content: str
    path: str = ""     # disk path — only set for CODE artifacts

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\n", " ")
        return f"Artifact(type={self.type!r}, title={self.title!r}, content={preview!r}...)"


@dataclass
class Aspect:
    """A tracked sub-task from the goal decomposition."""
    name: str
    status: str        # "pending" | "in_progress" | "complete" | "failed"

    @property
    def complete(self) -> bool:
        return self.status == "complete"

    def __repr__(self) -> str:
        return f"Aspect(name={self.name!r}, status={self.status!r})"


@dataclass
class GoalResult:
    """
    Structured result returned by Engine.run().

    Attributes
    ----------
    run_id      : unique identifier for this run
    status      : "completed" | "failed" | "cancelled"
    success     : True when all aspects complete and DONE fired with full reward
    steps       : number of steps taken
    reward      : cumulative shaped reward
    goal        : the original goal string
    artifacts   : all artifacts produced (CODE, DOCUMENT, NARRATIVE, ANALYSIS)
    aspects     : all tracked aspects and their final statuses
    log_path    : path to the step-by-step JSON log file
    raw         : the full raw result dict from the engine (escape hatch)
    """
    run_id    : str
    status    : str
    success   : bool
    steps     : int
    reward    : float
    goal      : str
    artifacts : list[Artifact] = field(default_factory=list)
    aspects   : list[Aspect]   = field(default_factory=list)
    log_path  : str = ""
    raw       : dict[str, Any] = field(default_factory=dict)

    # ── Convenience accessors ─────────────────────────────────────────────────

    def first(self, artifact_type: str) -> Artifact | None:
        """Return the first artifact of the given type, or None."""
        t = artifact_type.upper()
        return next((a for a in self.artifacts if a.type == t), None)

    def get(self, artifact_type: str) -> list[Artifact]:
        """Return all artifacts of the given type."""
        t = artifact_type.upper()
        return [a for a in self.artifacts if a.type == t]

    @property
    def code(self) -> str | None:
        """Content of the first CODE artifact, or None."""
        art = self.first("CODE")
        return art.content if art else None

    @property
    def document(self) -> str | None:
        """Content of the first DOCUMENT artifact, or None."""
        art = self.first("DOCUMENT")
        return art.content if art else None

    @property
    def narrative(self) -> str:
        """All NARRATIVE artifact content concatenated."""
        return "\n\n".join(a.content for a in self.get("NARRATIVE") if a.content)

    def aspect(self, name: str) -> Aspect | None:
        """Look up an aspect by name."""
        return next((a for a in self.aspects if a.name == name), None)

    def __repr__(self) -> str:
        return (
            f"GoalResult(status={self.status!r}, success={self.success}, "
            f"steps={self.steps}, reward={self.reward:.1f}, "
            f"artifacts={len(self.artifacts)}, aspects={len(self.aspects)})"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """
    Main entry point for running goals with OpenEngine.

    Credentials can be supplied explicitly (for multi-tenant / hosted use)
    or read from environment variables in .env (for local dev).  Explicit
    parameters always win over env vars.

    Parameters
    ----------
    llm_provider : str
        "openai" | "anthropic" | "gemini" | "mistral" | "cloudflare"
        Defaults to LLM_PROVIDER env var, then "cloudflare".
    llm_api_key : str
        API key for OpenAI / Anthropic / Gemini / Mistral.
        Defaults to LLM_API_KEY env var.
    llm_model : str
        Model ID override (e.g. "gpt-4o", "claude-opus-4-6").
        Defaults to LLM_MODEL env var, then the provider's recommended default.
    cf_account_id : str
        Cloudflare account ID — only required when llm_provider="cloudflare".
    cf_api_token : str
        Cloudflare API token — only required when llm_provider="cloudflare".
    capability_manifest_path : str
        Path to the JSON capability manifest that declares available tools.
    user_id : str
        Logical user identifier attached to every run.
    """

    def __init__(
        self,
        llm_provider             : str = "",
        llm_api_key              : str = "",
        llm_model                : str = "",
        cf_account_id            : str = "",
        cf_api_token             : str = "",
        tavily_api_key           : str = "",
        brave_api_key            : str = "",
        resend_api_key           : str = "",
        capability_manifest_path : str = "",
        user_id                  : str = "",
    ) -> None:
        self._llm_provider   = llm_provider
        self._llm_api_key    = llm_api_key
        self._llm_model      = llm_model
        self._cf_account_id  = cf_account_id
        self._cf_api_token   = cf_api_token
        self._tavily_api_key = tavily_api_key
        self._brave_api_key  = brave_api_key
        self._resend_api_key = resend_api_key
        self._manifest_path = (
            capability_manifest_path
            or os.getenv("XYBER_CAPABILITY_MANIFEST_PATH", "")
            or str(_REPO_ROOT / "examples" / "manifests" / "toolserver.json")
        )
        self._user_id = (
            user_id
            or os.getenv("XYBER_LOCAL_USER_ID", "sdk-user")
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        aspects: list[str] | None = None,
        run_id: str = "",
    ) -> GoalResult:
        """
        Run a goal to completion and return a structured GoalResult.

        Parameters
        ----------
        goal    : plain-English goal string
        aspects : optional list of pre-defined aspect names — when provided,
                  PLAN is skipped and these aspects are injected directly.
                  Last aspect should be 'email_transmission_confirmation' if
                  email delivery is required.
        run_id  : optional explicit run ID (auto-generated if omitted)

        Returns
        -------
        GoalResult
        """
        from engine.inference_worker import run_local_goal

        raw = run_local_goal(
            goal                     = goal,
            user_id                  = self._user_id,
            run_id                   = run_id,
            capability_manifest_path = self._manifest_path,
            pre_defined_aspects      = aspects,
            llm_provider             = self._llm_provider,
            llm_api_key              = self._llm_api_key,
            llm_model                = self._llm_model,
            cf_account_id            = self._cf_account_id,
            cf_api_token             = self._cf_api_token,
            tavily_api_key           = self._tavily_api_key,
            brave_api_key            = self._brave_api_key,
            resend_api_key           = self._resend_api_key,
        )
        return self._wrap(goal, raw)

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _wrap(goal: str, raw: dict[str, Any]) -> GoalResult:
        """Convert the raw result dict into a typed GoalResult."""
        artifacts = [
            Artifact(
                type    = a.get("artifact_type", ""),
                title   = a.get("title", ""),
                content = a.get("content") or "",
                path    = (a.get("metadata") or {}).get("script_path", "")
                          if isinstance(a.get("metadata"), dict) else "",
            )
            for a in (raw.get("artifacts") or [])
        ]

        aspects = [
            Aspect(
                name   = a.get("aspect", ""),
                status = a.get("status", "pending"),
            )
            for a in (raw.get("aspects") or [])
        ]

        return GoalResult(
            run_id    = raw.get("run_id", ""),
            status    = raw.get("status", "failed"),
            success   = bool(raw.get("success", False)),
            steps     = int(raw.get("step_count", 0)),
            reward    = float(raw.get("reward", 0.0)),
            goal      = goal,
            artifacts = artifacts,
            aspects   = aspects,
            log_path  = raw.get("log_path", ""),
            raw       = raw,
        )
