"""
tick_logger.py — Xybernetex Agent Telemetry Logger

Writes per-step tick JSON files and a final summary to a flat local directory.
Optionally uploads all files to OVH Object Storage (S3-compatible) in the
background via a daemon queue thread so the inference loop is never blocked.

Directory layout (always flat — no per-run subdirectories):
  Worker/logs/       {run_id}_step_{step:03d}.json
                     {run_id}_summary.json
  Worker/artifacts/  {run_id}_step_{step:03d}_{artifact_type}.{ext}

Environment variables:
  OBJECT_STORAGE_UPLOAD=true      (default false — local filesystem only)
  OVH_OS_ENDPOINT=https://...     S3-compatible endpoint URL (OVH Swift/S3)
  OVH_OS_ACCESS_KEY=...
  OVH_OS_SECRET_KEY=...
  OVH_OS_BUCKET=xybernetex-runs   (default)

Tick schema (per-step JSON):
  tick_type, run_id, step, timestamp, action {index, name},
  state_signals (key booleans + scalars extracted from 256-dim vector),
  logits (raw 6-dim float list), reward {step, cumulative, label},
  done, llm {prompt, response}, cognitive_stream (3-line list),
  aspects {before, after}, exec_result {success, returncode, stdout, stderr},
  artifact {type, title, content_preview, file_path}
"""
from __future__ import annotations

import json
import logging
import os
import queue as _queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("tick_logger")

# ── Directory roots ───────────────────────────────────────────────────────────

_HERE     = Path(__file__).resolve().parent
LOGS_DIR  = _HERE / "logs"
ARTS_DIR  = _HERE / "artifacts"

# Artifact-type → file extension
_EXTENSIONS: dict[str, str] = {
    "CODE"      : "py",
    "DOCUMENT"  : "md",
    "NARRATIVE" : "txt",
    "ANALYSIS"  : "txt",
    "DIAGRAM"   : "txt",
    "DECISION"  : "txt",
    "REPLAN"    : "txt",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ARTS_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ── OVH Object Storage uploader ───────────────────────────────────────────────

class _ObjectStorageUploader:
    """
    Daemon thread that drains an upload queue and pushes files to
    OVH Object Storage (S3-compatible endpoint) via boto3.

    Silently disables itself if:
      - OBJECT_STORAGE_UPLOAD != "true"
      - Required env vars are missing
      - boto3 is not installed
    The inference loop is never blocked or crashed by upload failures.
    """

    def __init__(self) -> None:
        self._enabled  : bool   = False
        self._client           = None
        self._bucket   : str    = ""
        self._q        : _queue.Queue[tuple[str, Path]] = _queue.Queue()
        self._thread           = threading.Thread(target=self._drain, daemon=True, name="xyber-uploader")
        self._started  : bool   = False
        self._setup()

    def _setup(self) -> None:
        if os.getenv("OBJECT_STORAGE_UPLOAD", "false").lower() != "true":
            return

        endpoint   = os.getenv("OVH_OS_ENDPOINT",   "").strip()
        access_key = os.getenv("OVH_OS_ACCESS_KEY",  "").strip()
        secret_key = os.getenv("OVH_OS_SECRET_KEY",  "").strip()
        bucket     = os.getenv("OVH_OS_BUCKET",      "xybernetex-runs").strip()

        if not all([endpoint, access_key, secret_key]):
            log.warning(
                "OBJECT_STORAGE_UPLOAD=true but OVH_OS_ENDPOINT / "
                "OVH_OS_ACCESS_KEY / OVH_OS_SECRET_KEY not all set. "
                "Object storage upload disabled."
            )
            return

        try:
            import boto3  # type: ignore
            self._client  = boto3.client(
                "s3",
                endpoint_url         = endpoint,
                aws_access_key_id    = access_key,
                aws_secret_access_key= secret_key,
            )
            self._bucket  = bucket
            self._enabled = True
            log.info("OVH Object Storage upload enabled → bucket=%s endpoint=%s", bucket, endpoint)
        except ImportError:
            log.warning(
                "boto3 is not installed — OBJECT_STORAGE_UPLOAD=true ignored. "
                "Run: pip install boto3"
            )

    def start(self) -> None:
        if self._enabled and not self._started:
            self._thread.start()
            self._started = True

    def enqueue(self, object_key: str, local_path: Path) -> None:
        """Enqueue a local file for background upload. No-op if disabled."""
        if self._enabled:
            self._q.put_nowait((object_key, local_path))

    def _drain(self) -> None:
        """Drain the upload queue indefinitely (daemon thread — dies with process)."""
        while True:
            try:
                object_key, local_path = self._q.get(timeout=5.0)
            except _queue.Empty:
                continue
            try:
                self._client.upload_file(str(local_path), self._bucket, object_key)
                log.debug("Uploaded %s → s3://%s/%s", local_path.name, self._bucket, object_key)
            except Exception as exc:
                log.warning("Upload failed for %s: %s", local_path.name, exc)
            finally:
                self._q.task_done()


# Module-level singleton — created once at import time
_uploader = _ObjectStorageUploader()


# ── State signal extractor ────────────────────────────────────────────────────

def extract_state_signals(state: "np.ndarray | list[float]") -> dict[str, float]:
    """
    Pull the human-readable key signals out of the 256-dim state vector.
    Index assignments must stay in sync with rl_chassis._build_structural().
    """
    def _s(i: int) -> float:
        try:
            return float(state[i])
        except (IndexError, TypeError):
            return 0.0

    return {
        # Expert mirror signals (s[10-16])
        "needs_planning"         : _s(10),
        "has_pending_aspects"    : _s(11),
        "all_aspects_complete"   : _s(12),
        "has_in_progress_aspects": _s(13),
        "has_unexecuted_code"    : _s(14),
        "mean_aspect_completion" : _s(15),
        "code_goal_bit"          : _s(16),
        # Step budget (s[40-41])
        "step_fraction"          : _s(40),
        "steps_remaining_frac"   : _s(41),
        # Error telemetry (s[42-44])
        "total_error_rate"       : _s(42),
        "unresolved_error_rate"  : _s(43),
        "error_resolution_rate"  : _s(44),
        # Execution quality (s[45-47])
        "exec_success_rate"      : _s(45),
        "exec_timeout_rate"      : _s(46),
        "any_exec_success_bit"   : _s(47),
        # Artifact distribution (s[48-56])
        "artifact_code_rate"     : _s(48),
        "artifact_document_rate" : _s(49),
        "artifact_narrative_rate": _s(50),
        "artifact_analysis_rate" : _s(51),
        "total_artifact_rate"    : _s(55),
        "duplicate_artifact_rate": _s(56),
        # AEU / safety / done-fail (s[60-63])
        "aeu_density"            : _s(60),
        "safety_event_rate"      : _s(61),
        "invalid_done_rate"      : _s(62),
        "local_goal_bit"         : _s(63),
    }


# ── TickLogger ────────────────────────────────────────────────────────────────

class TickLogger:
    """
    Per-run telemetry writer.

    Call write_step() after every env.step().
    Call write_artifact_file() whenever a CODE/DOCUMENT artifact is produced.
    Call write_summary() once at run end.

    All writes are immediate (local filesystem). Object storage uploads happen
    in a background daemon thread and do not block the inference loop.
    """

    def __init__(self, run_id: str, goal: str) -> None:
        _ensure_dirs()
        _uploader.start()
        self.run_id      = run_id
        self.goal        = goal
        self._step_ticks : list[dict[str, Any]] = []

    # ── per-step tick ─────────────────────────────────────────────────────────

    def write_step(self, data: dict[str, Any]) -> Path:
        """
        Write one step tick JSON immediately to logs/.
        data should be the full tick dict built by inference_worker.py.
        Returns the local path written.
        """
        step = int(data.get("step", 0))
        data["tick_type"] = "step"
        data["run_id"]    = self.run_id
        data.setdefault("timestamp", _utc_now())

        filename = f"{self.run_id}_step_{step:03d}.json"
        path     = LOGS_DIR / filename
        _write_json(path, data)
        _uploader.enqueue(filename, path)

        # Keep a reference for embedding in the summary
        self._step_ticks.append(data)
        return path

    # ── artifact content file ─────────────────────────────────────────────────

    def write_artifact_file(
        self,
        step          : int,
        artifact_type : str,
        content       : str,
        title         : str = "",
    ) -> Path:
        """
        Write raw artifact content to artifacts/ as a standalone file.
        Extension is determined by artifact_type (CODE→.py, DOCUMENT→.md, etc.).
        Returns the local path written.
        """
        ext      = _EXTENSIONS.get(artifact_type.upper(), "txt")
        slug     = artifact_type.lower()
        filename = f"{self.run_id}_step_{step:03d}_{slug}.{ext}"
        path     = ARTS_DIR / filename
        path.write_text(content, encoding="utf-8")
        _uploader.enqueue(filename, path)
        log.debug("Artifact file written: %s (%d chars)", filename, len(content))
        return path

    # ── final summary ─────────────────────────────────────────────────────────

    def write_summary(self, data: dict[str, Any]) -> Path:
        """
        Write the final roll-up JSON to logs/.
        All per-step ticks are embedded under 'step_ticks'.
        Uploaded last — acts as a completion sentinel in the bucket.
        """
        summary = {
            "tick_type"  : "summary",
            "run_id"     : self.run_id,
            "goal"       : self.goal,
            "timestamp"  : _utc_now(),
            **data,
            "step_ticks" : self._step_ticks,
        }
        filename = f"{self.run_id}_summary.json"
        path     = LOGS_DIR / filename
        _write_json(path, summary)
        _uploader.enqueue(filename, path)
        log.info(
            "Run summary written: %s  steps=%d  reward=%.4f  status=%s",
            filename,
            data.get("total_steps", 0),
            data.get("total_reward", 0.0),
            data.get("status", "unknown"),
        )
        return path
