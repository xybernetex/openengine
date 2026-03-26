"""
inference_worker.py - Xybernetex Engine live worker.

Runs as:
  - a long-lived Redis consumer for the API task queue, or
  - a local single-run CLI executor for Windows/dev workflows.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import torch
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Load env BEFORE any module that reads env vars (live_executor, connectors, etc.)
# Without this, _resolve_capability_manifest() runs before dotenv loads and
# XYBER_CAPABILITY_MANIFEST_PATH is never found → empty manifest → LLM only
# sees web_search → send_email is never selected for delivery aspects.
load_dotenv(_HERE / ".env",        override=False)
load_dotenv(_HERE.parent / ".env", override=False)

import terminal as T
from connectors import CapabilityManifest, N8NWebhookConnector, ToolRegistry
from rl_chassis import ActorCritic, ActionSpace, STATE_DIM, ACTION_DIM
from tick_logger import TickLogger, extract_state_signals
from xyber_env import XybernetexEnv

_OS = platform.system()

MODEL_PATH = str(Path(__file__).resolve().parent.parent / "models" / "policy_v3.pt")
MAX_STEPS = 25

# State vector index constants — must stay in sync with rl_chassis._build_structural()
_S_NEEDS_PLANNING    = 10   # s[10]: 1.0 if no aspects exist yet
_S_HAS_PENDING       = 11   # s[11]: 1.0 if any aspect is pending
_S_ALL_COMPLETE      = 12   # s[12]: 1.0 if all aspects are complete
_S_HAS_IN_PROGRESS   = 13   # s[13]: 1.0 if any aspect is in_progress
_S_UNEXECUTED_CODE   = 14   # s[14]: 1.0 if unexecuted CODE artifacts exist
_S_MEAN_COMPLETION   = 15   # s[15]: mean completion ratio across aspects
_S_CODE_GOAL         = 16   # s[16]: 1.0 for code goals, 0.0 for strategic/web
_S_TOOL_GOAL         = 17   # s[17]: 1.0 for tool goals, 0.0 for local/strategic
_S_DELIVERY_PENDING  = 18   # s[18]: 1.0 when ALL remaining pending aspects require INVOKE_TOOL
_S_SEARCH_GOAL       = 19   # s[19]: 1.0 if goal requires web search (independent of code_goal)
_S_SEARCH_DONE       = 30   # s[30]: 1.0 once a NARRATIVE artifact exists (web search has fired)

LOGS_DIR   = _HERE / "logs"
OUTPUT_DIR = _HERE / "output"

ACTION_MAP = {
    0: "PLAN",
    1: "ANALYZE",
    2: "INVOKE_TOOL",
    3: "EXECUTE_CODE",
    4: "WRITE_ARTIFACT",
    5: "DONE",
}
ACTION_MAP_REV = {v: k for k, v in ACTION_MAP.items()}


class CancelledRun(Exception):
    """Raised when the API requests cancellation for an active run."""


class JobSink(Protocol):
    def update_job(self, run_id: str, fields: dict[str, Any]) -> None: ...
    def publish_event(self, run_id: str, event_data: dict[str, Any]) -> None: ...
    def is_cancelled(self, run_id: str) -> bool: ...
    def store_result(
        self,
        run_id: str,
        conclusion: str,
        report_md: str,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None: ...


class LocalJobQueue:
    """
    Minimal local sink used for terminal-first development without Redis.

    Writes job state, event history, and final results under the run output
    directory so local runs remain inspectable.
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, dict[str, Any]] = {}

    def _run_dir(self, run_id: str) -> Path:
        path = self.base_dir / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _job_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "job.json"

    def _events_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "events.jsonl"

    def _result_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "result.json"

    def update_job(self, run_id: str, fields: dict[str, Any]) -> None:
        current = self._jobs.setdefault(run_id, {"run_id": run_id})
        current.update(fields)
        self._job_path(run_id).write_text(
            json.dumps(current, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def publish_event(self, run_id: str, event_data: dict[str, Any]) -> None:
        event_type = str(event_data.get("type", "") or "")
        payload = {
            "event": event_type,
            "run_id": run_id,
            "timestamp": _utc_now(),
            **event_data,
        }
        with self._events_path(run_id).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def is_cancelled(self, run_id: str) -> bool:
        return (self._run_dir(run_id) / "cancel.flag").exists()

    def store_result(
        self,
        run_id: str,
        conclusion: str,
        report_md: str,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> None:
        payload = {
            "conclusion": conclusion,
            "report_md": report_md,
            "artifacts": artifacts or [],
        }
        self._result_path(run_id).write_text(
            json.dumps(payload, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )

    def result_path(self, run_id: str) -> Path:
        return self._result_path(run_id)


def _load_json_object(path_str: str) -> dict[str, Any]:
    path = Path(path_str).expanduser()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Capability manifest at {path} must be a JSON object.")
    return payload


def _resolve_capability_manifest(
    job: dict[str, Any],
    *,
    run_id: str,
    user_id: str,
) -> CapabilityManifest:
    raw_manifest = job.get("capability_manifest")
    manifest: CapabilityManifest | None = None

    if isinstance(raw_manifest, CapabilityManifest):
        manifest = raw_manifest
    elif isinstance(raw_manifest, dict):
        manifest = CapabilityManifest.from_dict(raw_manifest)
    elif isinstance(raw_manifest, str) and raw_manifest.strip():
        manifest = CapabilityManifest.from_dict(json.loads(raw_manifest))

    if manifest is None:
        manifest_path = str(job.get("capability_manifest_path", "") or "").strip()
        if manifest_path:
            manifest = CapabilityManifest.from_dict(_load_json_object(manifest_path))

    if manifest is None and isinstance(job.get("tools"), list):
        manifest = CapabilityManifest.from_dict({
            "tools": job["tools"],
            "source": "job.tools",
        })

    if manifest is None:
        env_manifest_json = os.getenv("XYBER_CAPABILITY_MANIFEST_JSON", "").strip()
        if env_manifest_json:
            manifest = CapabilityManifest.from_dict(json.loads(env_manifest_json))

    if manifest is None:
        env_manifest_path = os.getenv("XYBER_CAPABILITY_MANIFEST_PATH", "").strip()
        if env_manifest_path:
            manifest = CapabilityManifest.from_dict(_load_json_object(env_manifest_path))

    if manifest is None:
        env_tools_json = os.getenv("XYBER_TOOLS_JSON", "").strip()
        if env_tools_json:
            manifest = CapabilityManifest.from_dict({
                "tools": json.loads(env_tools_json),
                "source": "env.XYBER_TOOLS_JSON",
            })

    if manifest is None:
        manifest = CapabilityManifest(source="worker_default")

    return manifest.bind(run_id=run_id, user_id=user_id)


def load_policy(device: torch.device) -> ActorCritic:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model weights not found: {MODEL_PATH}")

    policy = ActorCritic(STATE_DIM, ACTION_DIM).to(device)
    policy.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    policy.eval()
    return policy


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_result_payload(
    env: XybernetexEnv,
    run_id: str,
    goal: str,
    status: str,
    success: bool,
    step_count: int,
    total_reward: float,
    summary_text: str,
) -> tuple[str, str, list[dict[str, Any]]]:
    artifacts = env.wm.get_artifacts(run_id)
    aspect_rows = env.wm.get_goal_aspects(run_id)

    artifact_payload = [
        {
            "id": index + 1,
            "artifact_type": item.get("artifact_type", ""),
            "title": item.get("title", ""),
            "content": (item.get("content") or "")[:4096],
        }
        for index, item in enumerate(artifacts[:100])
    ]

    report_lines = [
        f"# Xybernetex RL Run",
        "",
        f"- Run ID: {run_id}",
        f"- Goal: {goal}",
        f"- Status: {status}",
        f"- Success: {'yes' if success else 'no'}",
        f"- Step Count: {step_count}",
        f"- Total Reward: {total_reward:.4f}",
        "",
        "## Summary",
        "",
        summary_text,
    ]
    if aspect_rows:
        report_lines.extend(["", "## Aspects", ""])
        for aspect in aspect_rows:
            report_lines.append(
                f"- {aspect.get('aspect', '')}: {aspect.get('status', 'pending')}"
            )
    if artifact_payload:
        report_lines.extend(["", "## Artifacts", ""])
        for artifact in artifact_payload:
            report_lines.append(
                f"- [{artifact['artifact_type']}] {artifact['title']}"
            )

    return summary_text, "\n".join(report_lines), artifact_payload


def _resolve_action(
    state, policy: ActorCritic, device: torch.device
) -> tuple[int, str, list[float]]:
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = policy(state_tensor)

    # Capture raw (pre-mask) logits for telemetry
    raw_logits: list[float] = logits[0].tolist()

    sv = state
    needs_planning      = sv[_S_NEEDS_PLANNING]     > 0.5
    all_complete        = sv[_S_ALL_COMPLETE]        > 0.5
    has_unexecuted_code = sv[_S_UNEXECUTED_CODE]    > 0.5
    is_tool_goal        = sv[_S_TOOL_GOAL]          > 0.5
    delivery_pending    = sv[_S_DELIVERY_PENDING]   > 0.5
    is_search_goal      = sv[_S_SEARCH_GOAL]        > 0.5
    search_done         = sv[_S_SEARCH_DONE]        > 0.5

    mask = torch.ones(ACTION_DIM, dtype=torch.bool)

    if delivery_pending:
        # s[18]=1.0: every remaining pending aspect is a delivery gate.
        # The ONLY valid action is INVOKE_TOOL — override everything else.
        mask[:] = False
        mask[ACTION_MAP_REV["INVOKE_TOOL"]] = True
    elif needs_planning:
        mask[:] = False
        mask[ACTION_MAP_REV["PLAN"]] = True
    elif is_search_goal and not search_done:
        # s[19]=1.0 and no NARRATIVE yet: goal requires web search and it hasn't
        # happened yet.  Force INVOKE_TOOL so the policy can't skip straight to
        # WRITE_ARTIFACT and hallucinate research data.
        mask[:] = False
        mask[ACTION_MAP_REV["INVOKE_TOOL"]] = True
    else:
        if not all_complete:
            mask[ACTION_MAP_REV["DONE"]] = False
        if not has_unexecuted_code:
            mask[ACTION_MAP_REV["EXECUTE_CODE"]] = False
        if not is_tool_goal:
            # INVOKE_TOOL only re-opens via delivery_pending (handled above).
            mask[ACTION_MAP_REV["INVOKE_TOOL"]] = False
        mask[ACTION_MAP_REV["PLAN"]] = False

    logits[0, ~mask] = float("-inf")
    action_idx = torch.argmax(logits, dim=-1).item()
    return action_idx, ACTION_MAP.get(action_idx, "UNKNOWN"), raw_logits


def _reward_label(r: float) -> str:
    """Human-readable bucket for a step reward — purely for telemetry readability."""
    if r >= 10.0:  return "terminal_success"
    if r >= 0.8:   return "aspect_complete"
    if r >= 0.3:   return "exec_success"
    if r > -0.2:   return "step_penalty"
    if r > -1.0:   return "stagnation"
    if r <= -4.0:  return "done_invalid"
    return "penalty"


def run_inference_job(
    policy: ActorCritic,
    queue: JobSink,
    job: dict[str, Any],
) -> dict[str, Any]:
    goal = job.get("goal", "").strip()
    user_id = str(job.get("user_id", "") or "")
    run_id = str(job.get("run_id", "") or "")

    if not goal or not run_id:
        raise ValueError("Dequeued job is missing required goal or run_id.")

    capability_manifest = _resolve_capability_manifest(
        job,
        run_id=run_id,
        user_id=user_id,
    )

    queue.update_job(run_id, {
        "status": "running",
        "goal": goal,
        "user_id": user_id,
        "worker": "xyber-v2",
        "started_at": _utc_now(),
        "capability_manifest": capability_manifest.to_dict(),
        "tool_count": len(capability_manifest.tools),
    })

    env = XybernetexEnv(
        mock_llm       = False,
        memory_backend = "sqlite",
        output_dir     = OUTPUT_DIR / run_id,
        user_id        = user_id,
        llm_provider   = job.get("llm_provider",   ""),
        llm_api_key    = job.get("llm_api_key",    ""),
        llm_model      = job.get("llm_model",      ""),
        cf_account_id  = job.get("cf_account_id",  ""),
        cf_api_token   = job.get("cf_api_token",   ""),
        tavily_api_key = job.get("tavily_api_key", ""),
        brave_api_key  = job.get("brave_api_key",  ""),
        resend_api_key = job.get("resend_api_key", ""),
    )

    T.set_event_sink(queue.publish_event, run_id)
    try:
        if hasattr(env, "executor") and env.executor is not None:
            if hasattr(env.executor, "set_execution_streamer"):
                env.executor.set_execution_streamer(T.stream_execution_live)
            if hasattr(env.executor, "set_event_publisher"):
                env.executor.set_event_publisher(queue.publish_event)
            if hasattr(env.executor, "set_tool_registry"):
                env.executor.set_tool_registry(ToolRegistry.from_manifest(capability_manifest))
            if (
                hasattr(env.executor, "set_tool_gateway")
                and os.getenv("XYBER_TOOL_WEBHOOK_URL", "").strip()
            ):
                env.executor.set_tool_gateway(N8NWebhookConnector())

        state = env.reset(
            goal=goal,
            user_id=user_id,
            run_id=run_id,
            capability_manifest=capability_manifest.to_dict(),
        )
        actual_run_id = env.current_run_id

        pre_defined_aspects = job.get("pre_defined_aspects") or []
        if pre_defined_aspects:
            clean = [
                re.sub(r"[^a-z0-9_]", "_", a.strip().lower())
                for a in pre_defined_aspects
                if isinstance(a, str) and len(a.strip()) > 2
            ][:5]
            for asp in clean:
                env.wm.upsert_goal_aspect(actual_run_id, asp, "pending", 0)
            T.write(
                f"  {T.CYAN}Workflow mode — {len(clean)} aspects pre-loaded, "
                f"PLAN skipped.{T.RESET}"
            )
            state = env._get_obs()

        tick_logger = TickLogger(actual_run_id, goal)

        T.boot_sequence(goal, MODEL_PATH, run_id=actual_run_id)
        T.bridge.register(env.wm, actual_run_id)

        queue.publish_event(actual_run_id, {
            "type": "narrative",
            "data": {"text": f"Run accepted for user {user_id or 'anonymous'}."},
        })

        done = False
        step = 0
        cumulative_reward = 0.0
        prev_reward = None
        step_log: list[dict[str, Any]] = []
        last_info: dict[str, Any] = {}

        cf_account_id = os.getenv("CF_ACCOUNT_ID", "")
        cf_api_token = os.getenv("CF_API_TOKEN", "")

        while not done and step < MAX_STEPS:
            if queue.is_cancelled(actual_run_id):
                raise CancelledRun(f"Run {actual_run_id} cancelled before step {step + 1}.")

            step += 1
            aspects_before = env.wm.get_goal_aspects(actual_run_id)

            action_idx, action_name, raw_logits = _resolve_action(
                state, policy, torch.device("cpu")
            )

            T.step_announce(step, MAX_STEPS, action_name, prev_reward=prev_reward)

            cog_stream = None
            if cf_account_id and cf_api_token:
                cog_stream = T.CognitiveStream(
                    cf_account_id,
                    cf_api_token,
                    goal,
                    action_name,
                )
                cog_stream.start()

            reasoning = T.LiveReasoning(action_name, actual_run_id, cognitive_stream=cog_stream)
            reasoning.start()

            state, reward, done, info = env.step(action_idx)

            reasoning.stop()
            cog_lines: list[str] = []
            if cog_stream:
                cog_stream.stop()
                cog_lines = cog_stream.get_lines()

            cumulative_reward += reward
            prev_reward = reward
            last_info = info or {}

            T.reveal_reward(reward, step, cumulative_reward)

            # ── Capture LLM telemetry from the executor ──────────────────────
            executor = getattr(env, "executor", None)
            llm_prompt   = getattr(executor, "_last_prompt",   "") if executor else ""
            llm_response = getattr(executor, "_last_response", "") if executor else ""
            exec_result  = getattr(executor, "_last_result",   {}) if executor else {}

            aspects_after = env.wm.get_goal_aspects(actual_run_id)

            # ── Write artifact content file if one was produced ───────────────
            artifact_type = exec_result.get("artifact_type", "")
            artifact_title = ""
            if artifact_type:
                # Fetch the content of the most recent artifact of this type
                artifacts_now = env.wm.get_artifacts(actual_run_id)
                latest = next(
                    (a for a in reversed(artifacts_now)
                     if a.get("artifact_type") == artifact_type),
                    None,
                )
                if latest:
                    artifact_title   = latest.get("title", "")
                    artifact_content = latest.get("content") or ""
                    tick_logger.write_artifact_file(
                        step, artifact_type, artifact_content, title=artifact_title
                    )

            # ── Build and write the full step tick ────────────────────────────
            tick: dict[str, Any] = {
                "step"      : step,
                "timestamp" : _utc_now(),
                "action"    : {"index": action_idx, "name": action_name},
                "state_signals": extract_state_signals(state),
                "logits"    : raw_logits,
                "reward"    : {
                    "step"       : reward,
                    "cumulative" : cumulative_reward,
                    "label"      : _reward_label(reward),
                },
                "done"      : done,
                "llm"       : {"prompt": llm_prompt, "response": llm_response},
                "cognitive_stream": cog_lines,
                "aspects"   : {"before": aspects_before, "after": aspects_after},
                "exec_result": {
                    "success"    : last_info.get("exec_success", False),
                    "returncode" : last_info.get("exec_returncode", -1),
                    "stdout"     : (last_info.get("stdout") or "")[:4096],
                    "stderr"     : (last_info.get("stderr") or "")[:1024],
                },
                "artifact"  : {
                    "type"            : artifact_type,
                    "title"           : artifact_title,
                    "content_preview" : (
                        (exec_result.get("content") or "")[:500]
                        if not artifact_type else ""
                    ),
                },
            }
            tick_path = tick_logger.write_step(tick)
            T.write(f"  {T.GRAY}tick -> {tick_path}{T.RESET}")

            step_log.append({
                "step"       : step,
                "action"     : action_name,
                "reward"     : reward,
                "cumulative" : cumulative_reward,
                "done"       : done,
            })

            queue.update_job(actual_run_id, {
                "status"       : "running",
                "step_count"   : step,
                "last_action"  : action_name,
                "total_reward" : cumulative_reward,
            })

        success = done and prev_reward is not None and prev_reward >= 10.0
        final_status = "completed" if success else "failed"

        log_path = tick_logger.write_summary({
            "goal"          : goal,
            "user_id"       : user_id,
            "total_steps"   : step,
            "total_reward"  : cumulative_reward,
            "success"       : success,
            "status"        : final_status,
            "os"            : _OS,
            "model"         : MODEL_PATH,
            "steps"         : step_log,
        })
        T.run_complete(step, cumulative_reward, success, actual_run_id, log_path, goal)

        if not done:
            T.write(
                f"  {T.AMBER}[Warning] MAX_STEPS ({MAX_STEPS}) reached without DONE{T.RESET}"
            )

        queue.update_job(actual_run_id, {
            "status": final_status,
            "step_count": step,
            "total_reward": cumulative_reward,
            "success": str(success).lower(),
            "completed_at": _utc_now(),
            "log_path": str(log_path) if log_path else "",
            "result": "Goal achieved." if success else "Run ended without satisfying DONE gate.",
            "error": "" if success else last_info.get("error", ""),
        })
        summary_text = (
            "Run completed successfully."
            if success else
            "Run ended without satisfying the terminal condition."
        )
        conclusion, report_md, artifacts = _build_result_payload(
            env,
            actual_run_id,
            goal,
            final_status,
            success,
            step,
            cumulative_reward,
            summary_text,
        )
        queue.store_result(actual_run_id, conclusion, report_md, artifacts)
        queue.publish_event(actual_run_id, {
            "type": "artifact",
            "data": {
                "title": "RL Run Summary",
                "preview": conclusion[:500],
                "conclusion": conclusion,
            },
        })
        queue.publish_event(actual_run_id, {
            "type": "narrative",
            "data": {"text": summary_text},
        })
        queue.publish_event(actual_run_id, {
            "type": final_status,
            "data": {
                "conclusion": conclusion,
                "step_count": step,
                "total_reward": cumulative_reward,
            },
        })

        return {
            "run_id"      : actual_run_id,
            "status"      : final_status,
            "step_count"  : step,
            "success"     : success,
            "reward"      : cumulative_reward,
            "log_path"    : str(log_path) if log_path else "",
            "artifacts"   : artifacts,
            "aspects"     : env.wm.get_goal_aspects(actual_run_id),
        }

    except CancelledRun as exc:
        queue.update_job(actual_run_id, {
            "status": "cancelled",
            "step_count": env.step_count,
            "completed_at": _utc_now(),
            "error": str(exc),
            "result": "Run cancelled by user request.",
        })
        conclusion, report_md, artifacts = _build_result_payload(
            env,
            actual_run_id,
            goal,
            "cancelled",
            False,
            env.step_count,
            0.0,
            "Run cancelled by user request.",
        )
        queue.publish_event(actual_run_id, {
            "type": "artifact",
            "data": {
                "title": "RL Run Summary",
                "preview": conclusion[:500],
                "conclusion": conclusion,
            },
        })
        queue.publish_event(actual_run_id, {
            "type": "narrative",
            "data": {"text": "Run cancelled by user request."},
        })
        queue.publish_event(actual_run_id, {
            "type": "cancelled",
            "data": {
                "conclusion": conclusion,
                "step_count": env.step_count,
            },
        })
        return {
            "run_id": actual_run_id,
            "status": "cancelled",
            "step_count": env.step_count,
            "success": False,
        }
    except Exception as exc:
        T.write(f"\n  {T.RED}[Fatal] {exc}{T.RESET}")
        traceback.print_exc()
        queue.update_job(actual_run_id, {
            "status": "failed",
            "step_count": env.step_count,
            "completed_at": _utc_now(),
            "error": str(exc),
        })
        conclusion, report_md, artifacts = _build_result_payload(
            env,
            actual_run_id,
            goal,
            "failed",
            False,
            env.step_count,
            0.0,
            f"Worker failure: {exc}",
        )
        queue.store_result(actual_run_id, conclusion, report_md, artifacts)
        queue.publish_event(actual_run_id, {
            "type": "artifact",
            "data": {
                "title": "RL Run Summary",
                "preview": conclusion[:500],
                "conclusion": conclusion,
            },
        })
        queue.publish_event(actual_run_id, {
            "type": "narrative",
            "data": {"text": f"Worker failure: {exc}"},
        })
        queue.publish_event(actual_run_id, {
            "type": "failed",
            "data": {
                "error": str(exc),
                "conclusion": conclusion,
                "step_count": env.step_count,
            },
        })
        return {
            "run_id": actual_run_id,
            "status": "failed",
            "step_count": env.step_count,
            "success": False,
            "error": str(exc),
        }
    finally:
        env.close()
        T.clear_event_sink()
        T.blank()
        T.write(f"  {T.GRAY}Environment closed.{T.RESET}")
        T.blank()


def run_worker_loop() -> None:
    from job_queue import JobQueue

    device = torch.device("cpu")
    policy = load_policy(device)
    queue = JobQueue()

    T.write(f"  {T.CYAN}Worker online. Listening on {queue.queue_key}.{T.RESET}")
    while True:
        job = queue.dequeue()
        if not job:
            continue
        run_id = str(job.get("run_id", "") or "")
        T.write(f"  {T.GRAY}Dequeued run {run_id}.{T.RESET}")
        run_inference_job(policy, queue, job)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Xybernetex either as a long-lived Redis worker (no goal "
            "argument) or as a single local CLI run (with a goal argument)."
        )
    )
    parser.add_argument(
        "goal",
        nargs="?",
        help="Run one local inference job directly from the terminal.",
    )
    parser.add_argument(
        "--user-id",
        default=os.getenv("XYBER_LOCAL_USER_ID", "local-dev"),
        help="Logical user_id to associate with a local CLI run.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional explicit run_id for a local CLI run.",
    )
    parser.add_argument(
        "--capability-manifest",
        default="",
        help="Optional path to a JSON capability manifest for a local CLI run.",
    )
    parser.add_argument(
        "--aspects",
        default="",
        help=(
            "Comma-separated list of pre-defined aspects to inject, skipping PLAN. "
            "Example: 'research_data,python_report,email_transmission_confirmation'"
        ),
    )
    return parser.parse_args(argv)


def run_local_goal(
    goal: str,
    user_id: str = "local-dev",
    run_id: str = "",
    capability_manifest_path: str = "",
    pre_defined_aspects: list[str] | None = None,
    llm_provider   : str = "",
    llm_api_key    : str = "",
    llm_model      : str = "",
    cf_account_id  : str = "",
    cf_api_token   : str = "",
    tavily_api_key : str = "",
    brave_api_key  : str = "",
    resend_api_key : str = "",
) -> dict[str, Any]:
    resolved_goal = goal.strip()
    if not resolved_goal:
        raise ValueError("Local CLI goal cannot be empty.")

    local_run_id = run_id.strip() or f"local_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    queue = LocalJobQueue(OUTPUT_DIR)
    device = torch.device("cpu")
    policy = load_policy(device)

    T.write(f"  {T.CYAN}Local mode active. Redis is not required.{T.RESET}")
    result = run_inference_job(policy, queue, {
        "run_id"                  : local_run_id,
        "goal"                    : resolved_goal,
        "user_id"                 : user_id,
        "model"                   : "xyber-v2",
        "status"                  : "queued",
        "created_at"              : _utc_now(),
        "capability_manifest_path": capability_manifest_path.strip(),
        "pre_defined_aspects"     : pre_defined_aspects or [],
        "llm_provider"            : llm_provider,
        "llm_api_key"             : llm_api_key,
        "llm_model"               : llm_model,
        "cf_account_id"           : cf_account_id,
        "cf_api_token"            : cf_api_token,
        "tavily_api_key"          : tavily_api_key,
        "brave_api_key"           : brave_api_key,
        "resend_api_key"          : resend_api_key,
    })
    T.write(f"  {T.GRAY}Local result -> {queue.result_path(local_run_id)}{T.RESET}")
    return result


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    if args.goal:
        aspects = (
            [a.strip() for a in args.aspects.split(",") if a.strip()]
            if args.aspects else None
        )
        run_local_goal(
            args.goal,
            user_id=args.user_id,
            run_id=args.run_id,
            capability_manifest_path=args.capability_manifest,
            pre_defined_aspects=aspects,
        )
    else:
        run_worker_loop()
