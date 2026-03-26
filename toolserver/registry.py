"""
registry.py — ToolServer plugin registry

Auto-discovers tool modules from the tools/ directory.
Each module must export:

    async def run(invocation: dict) -> dict

The returned dict must contain:
    {
        "status":  "succeeded" | "failed" | "timed_out",
        "output":  {...},   # arbitrary result data
        "error":   "...",   # empty string on success
    }

The registry wraps this in the full ToolResult envelope expected by the Worker.
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("tool_server.registry")

_TOOLS_DIR = Path(__file__).resolve().parent / "tools"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_result(
    invocation: dict[str, Any],
    *,
    status: str,
    output: dict[str, Any],
    error: str = "",
) -> dict[str, Any]:
    """Build a ToolResult-compatible response envelope."""
    return {
        "invocation_id": invocation.get("invocation_id", ""),
        "run_id":        invocation.get("run_id", ""),
        "status":        status,
        "output":        output,
        "error":         error,
        "completed_at":  _utc_now(),
    }


class ToolRegistry:
    """
    Loads tool modules from tools/ and dispatches invocations to them.

    Tool selection: invocation["tool_name"] must match a discovered module name.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Any] = {}  # tool_name -> async run() fn

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self) -> None:
        """
        Scan tools/ for Python modules and import any that export run().
        Called once at startup.
        """
        if not _TOOLS_DIR.exists():
            logger.warning("tools/ directory not found at %s", _TOOLS_DIR)
            return

        for path in sorted(_TOOLS_DIR.glob("*.py")):
            if path.name.startswith("_"):
                continue
            tool_name = path.stem  # e.g. web_search

            try:
                spec   = importlib.util.spec_from_file_location(
                    f"tools.{tool_name}", path
                )
                module = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
                spec.loader.exec_module(module)                       # type: ignore[union-attr]

                run_fn = getattr(module, "run", None)
                if run_fn is None or not callable(run_fn):
                    logger.warning(
                        "Skipping %s: no callable run() exported", path.name
                    )
                    continue

                self._handlers[tool_name] = run_fn
                logger.info("Registered tool: %s", tool_name)

            except Exception:
                logger.error(
                    "Failed to load tool %s:\n%s", path.name, traceback.format_exc()
                )

    # ── Invocation ────────────────────────────────────────────────────────────

    async def invoke(self, invocation: dict[str, Any]) -> dict[str, Any]:
        tool_name: str = str(invocation.get("tool_name", "")).strip()

        handler = self._handlers.get(tool_name)
        if handler is None:
            return _make_result(
                invocation,
                status="failed",
                output={},
                error=f"tool_not_found: {tool_name!r}. "
                      f"Available: {list(self._handlers.keys())}",
            )

        try:
            # Support both sync and async handlers
            if inspect.iscoroutinefunction(handler):
                result = await handler(invocation)
            else:
                result = handler(invocation)

            if not isinstance(result, dict):
                raise TypeError(
                    f"Tool {tool_name}.run() must return a dict, got {type(result)}"
                )

            # Normalise — ensure required fields present
            status = str(result.get("status", "failed"))
            if status not in {"succeeded", "failed", "timed_out"}:
                status = "failed"

            output = result.get("output", {})
            if not isinstance(output, dict):
                output = {"raw": str(output)}

            return _make_result(
                invocation,
                status=status,
                output=output,
                error=str(result.get("error", "")),
            )

        except Exception:
            tb = traceback.format_exc()
            logger.error("Tool %s raised an exception:\n%s", tool_name, tb)
            return _make_result(
                invocation,
                status="failed",
                output={},
                error=f"tool_exception: {tb[-400:]}",
            )

    # ── Introspection ─────────────────────────────────────────────────────────

    def tool_names(self) -> list[str]:
        return sorted(self._handlers.keys())

    def is_registered(self, name: str) -> bool:
        return name in self._handlers
