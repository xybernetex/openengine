"""
live_executor.py — Xybernetex Live Action Dispatcher

Bridges the PPO policy's discrete action decisions to real module calls.
Each handler is responsible for:
  1. Calling the appropriate backend (LLM / webhook tools / code_executor)
  2. Persisting results to WorkingMemory
  3. Returning a standardised result dict that xyber_env.py uses to
     populate TransitionInfo for reward calculation.

LLM calls are routed through LLMClient (llm_client.py), which selects the
provider from the LLM_PROVIDER env var (cloudflare | openai | mistral |
anthropic | gemini).  No provider-specific code lives here.

Handler map
-----------
  PLAN          -> LLM -> goal decomposition -> aspects in working memory
  ANALYZE       -> LLM -> ANALYSIS artifact
  INVOKE_TOOL   -> LLM picks tool from manifest -> webhook call (or web
                   search fallback) -> LLM synthesis -> NARRATIVE artifact
  WRITE_ARTIFACT -> LLM -> DOCUMENT or CODE artifact (CODE written to disk)
  EXECUTE_CODE  -> code_executor.run_code -> execution record + NARRATIVE
  DONE / others -> no-op (handled by xyber_env termination gate)
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import requests
from dotenv import load_dotenv

from llm_client import LLMClient
from connectors import (
    CapabilityManifest,
    ToolGateway,
    ToolInvocation,
    ToolRegistry,
    ToolResult,
)

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent

# Load env — try Reinforcement/.env then project root .env
load_dotenv(_HERE / ".env",        override=False)
load_dotenv(_HERE.parent / ".env", override=False)

if not os.getenv("XYBER_TOOL_WEBHOOK_URL", "").strip():
    logger.warning(
        "XYBER_TOOL_WEBHOOK_URL is not set. INVOKE_TOOL will fall back to "
        "web search only — no N8N webhook calls will be made. "
        "Set XYBER_TOOL_WEBHOOK_URL in your .env to enable tool execution."
    )

# ── Code-signal vocabulary for artifact type detection ────────────────────────
_CODE_SIGNALS = frozenset({
    "script", "code", "function", "class", "calculate", "fibonacci",
    "sort", "parse", "generate", "implement", "program", "algorithm",
    "execute", "run", "automate", "compute", "build",
})

# ── Tool-locked aspect keywords ────────────────────────────────────────────────
# Aspects whose names contain any of these keywords can ONLY be marked complete
# by a successful INVOKE_TOOL call.  WRITE_ARTIFACT is blocked from completing
# them — prevents the LLM from writing "Email sent: confirmed" and self-awarding
# the completion reward without the tool actually being called.
_TOOL_LOCKED_KEYWORDS = frozenset({
    "email", "send", "transmission", "notification",
    "dispatch", "deliver", "notify", "sms", "push",
})

# ── Result dict keys (contract with xyber_env.TransitionInfo) ─────────────────
# success       : bool  — handler ran without hard failure
# exec_success  : bool  — code executed successfully (EXECUTE_CODE only)
# exec_returncode: int  — process exit code          (EXECUTE_CODE only)
# error         : str   — error message if success=False


class LiveExecutor:
    """
    Stateless action dispatcher.  One instance per XybernetexEnv episode set.

    Parameters
    ----------
    wm         : WorkingMemory instance (sqlite or pg)
    output_dir : Directory where CODE artifacts are written to disk
    """

    def __init__(
        self,
        wm        : Any,
        output_dir: Path,
        llm_provider  : str = "",
        llm_api_key   : str = "",
        llm_model     : str = "",
        cf_account_id : str = "",
        cf_api_token  : str = "",
        tavily_api_key: str = "",
        brave_api_key : str = "",
        resend_api_key: str = "",
    ) -> None:
        self.wm         = wm
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.llm        = LLMClient(
            provider   = llm_provider,
            api_key    = llm_api_key,
            model      = llm_model,
            cf_account = cf_account_id,
            cf_token   = cf_api_token,
        )
        self._tavily_api_key = tavily_api_key
        self._brave_api_key  = brave_api_key
        self._resend_api_key = resend_api_key
        self._execution_streamer = None  # optional fn(script_path, timeout) -> dict
        self._event_publisher = None     # optional fn(run_id, event_data) -> None
        self._tool_gateway: ToolGateway | None = None
        self._tool_registry = ToolRegistry()
        self._capability_manifest: CapabilityManifest | None = None
        # Telemetry capture — read by TickLogger after each dispatch() call
        self._last_prompt   : str            = ""
        self._last_response : str            = ""
        self._last_result   : dict[str, Any] = {}

    def _is_tool_locked(self, aspect_name: str) -> bool:
        """
        Return True if this aspect can only be marked complete by a successful
        INVOKE_TOOL call.  WRITE_ARTIFACT is blocked from completing it.

        Only the canonical delivery aspect name triggers the lock.  Keyword
        scanning is intentionally NOT used here because the PLAN LLM sometimes
        produces aspects like 'draft_email_summary' or 'send_findings' that
        contain delivery keywords but are pure content tasks — scanning those
        as tool-locked causes multiple send_email calls per run.
        """
        _TOOL_LOCKED_EXACT = frozenset({
            "email_transmission_confirmation",
            "sms_delivery_confirmation",
            "push_notification_confirmation",
            "webhook_delivery_confirmation",
        })
        return aspect_name.lower().strip() in _TOOL_LOCKED_EXACT

    def set_execution_streamer(self, fn) -> None:
        """Wire terminal.stream_execution_live (or any compatible callable) for real-time output."""
        self._execution_streamer = fn

    def set_event_publisher(self, fn) -> None:
        """Wire JobQueue.publish_event (or any compatible callable)."""
        self._event_publisher = fn

    def set_tool_gateway(self, gateway: ToolGateway | None) -> None:
        """Wire a stable connector runtime for external world actions."""
        self._tool_gateway = gateway

    def set_tool_registry(self, registry: ToolRegistry | None) -> None:
        """Wire the runtime's known connector catalog."""
        self._tool_registry = registry or ToolRegistry()

    def set_capability_manifest(
        self,
        manifest: CapabilityManifest | dict[str, Any] | None,
    ) -> None:
        """Wire the per-run allowed tool manifest used for planning and enforcement."""
        if manifest is None:
            self._capability_manifest = None
            return
        self._capability_manifest = (
            manifest
            if isinstance(manifest, CapabilityManifest)
            else CapabilityManifest.from_dict(manifest)
        )

    def _publish_event(self, run_id: str, event_data: dict[str, Any]) -> None:
        if self._event_publisher is None:
            return
        try:
            self._event_publisher(run_id, event_data)
        except Exception as e:
            logger.warning("Event publisher failed for run %s: %s", run_id, e)

    def _publish_narrative(self, run_id: str, text: str) -> None:
        if text:
            self._publish_event(run_id, {
                "type": "narrative",
                "data": {"text": text},
            })

    def _publish_aspects(self, run_id: str) -> None:
        self._publish_event(run_id, {
            "type": "aspect_update",
            "data": {"aspects": self.wm.get_goal_aspects(run_id)},
        })

    def _publish_tool_call(self, invocation: ToolInvocation) -> None:
        self._publish_event(invocation.run_id, {
            "type": "tool_call",
            "data": {
                "invocation_id": invocation.invocation_id,
                "tool_name": invocation.tool_name,
                "action": invocation.action,
                "status": "started",
            },
        })

    def _publish_tool_result(
        self,
        invocation: ToolInvocation,
        result: ToolResult,
    ) -> None:
        self._publish_event(invocation.run_id, {
            "type": "tool_result",
            "data": {
                "invocation_id": result.invocation_id,
                "tool_name": invocation.tool_name,
                "status": result.status,
                "output": result.output,
                "error": result.error,
            },
        })

    def _get_capability_manifest(self, run_id: str) -> CapabilityManifest | None:
        run = self.wm.get_run(run_id)
        user_id = str(run.get("user_id", "") or "")

        if self._capability_manifest is not None:
            if self._capability_manifest.run_id == run_id:
                return self._capability_manifest
            if not self._capability_manifest.run_id:
                self._capability_manifest = self._capability_manifest.bind(
                    run_id=run_id,
                    user_id=user_id,
                )
                return self._capability_manifest

        stored = None
        if hasattr(self.wm, "get_capability_manifest"):
            stored = self.wm.get_capability_manifest(run_id)
        if isinstance(stored, dict):
            manifest = CapabilityManifest.from_dict(stored)
            if manifest.run_id != run_id or manifest.user_id != user_id:
                manifest = manifest.bind(run_id=run_id, user_id=user_id)
            self._capability_manifest = manifest
            return manifest
        return None

    def _tool_context_lines(self, run_id: str) -> list[str]:
        manifest = self._get_capability_manifest(run_id)
        if manifest is None:
            return []

        lines = ["AVAILABLE TOOLS:"]
        if not manifest.tools:
            lines.append("  (none configured for this run)")
            return lines

        for tool in manifest.tools:
            actions = ", ".join(tool.actions) if tool.actions else "any action"
            access = "read-only" if tool.read_only else "mutating"
            lines.append(f"  [{tool.risk} / {access}] {tool.name} -> {actions}")
            if tool.description:
                lines.append(f"    {tool.description[:180]}")
        return lines

    def _check_tool_access(self, run_id: str, tool_name: str, action: str) -> str | None:
        manifest = self._get_capability_manifest(run_id)
        if manifest is None:
            return "capability_manifest_missing"
        descriptor = manifest.get_tool(tool_name)
        if descriptor is None:
            return f"tool_not_allowed: {tool_name}"
        if not descriptor.supports_action(action):
            return f"action_not_allowed: {tool_name}.{action}"
        if not self._tool_registry.is_registered(tool_name):
            return f"tool_not_registered: {tool_name}"
        return None

    def _invoke_tool(
        self,
        *,
        run_id: str,
        step_id: str,
        tool_name: str,
        action: str,
        payload: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Execute one external-world action through the connector runtime.

        This helper is intentionally backend-agnostic: the executor speaks in
        stable tool contracts, while the current runtime may be backed by n8n.
        """
        user_id = str(self.wm.get_run(run_id).get("user_id", "") or "")
        invocation = ToolInvocation.create(
            run_id=run_id,
            step_id=step_id,
            user_id=user_id,
            tool_name=tool_name,
            action=action,
            input=payload or {},
        )
        self._publish_tool_call(invocation)

        access_error = self._check_tool_access(run_id, tool_name, action)
        if access_error is not None:
            result = ToolResult.failure(
                invocation,
                access_error,
            )
        elif self._tool_gateway is None:
            result = ToolResult.failure(
                invocation,
                "tool_gateway_unconfigured",
            )
        else:
            result = self._tool_gateway.invoke(invocation)

        self._publish_tool_result(invocation, result)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC DISPATCH
    # ══════════════════════════════════════════════════════════════════════════

    def dispatch(
        self,
        action_name : str,
        run_id      : str,
        goal        : str,
        step_number : int,
    ) -> dict[str, Any]:
        """
        Route action_name to its handler.

        Always returns a dict — never raises.  Errors are captured in the
        'error' key and logged to stdout for the inference trace.
        """
        handlers = {
            "PLAN"          : self._handle_plan,
            "ANALYZE"       : self._handle_analyze,
            "INVOKE_TOOL"   : self._handle_invoke_tool,
            "WRITE_ARTIFACT": self._handle_write_artifact,
            "EXECUTE_CODE"  : self._handle_execute_code,
        }
        handler = handlers.get(action_name)
        if handler is None:
            # DONE, unknown — nothing to execute; termination gate handles DONE
            return {"success": True, "exec_success": False, "exec_returncode": -1}

        try:
            result = handler(run_id, goal, step_number)
        except Exception as exc:  # noqa: BLE001
            print(f"[LiveExecutor] {action_name} handler error: {exc}", flush=True)
            result = {"success": False, "error": str(exc),
                      "exec_success": False, "exec_returncode": -1}
        self._last_result = result
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # LLM CLIENT WRAPPER
    # ══════════════════════════════════════════════════════════════════════════

    def _llm_call(
        self,
        messages    : list[dict[str, str]],
        max_tokens  : int   = 2048,
        temperature : float = 0.5,
    ) -> str:
        """
        Single LLM call via LLMClient.

        Captures the prompt/response for TickLogger telemetry.  Never raises —
        returns a descriptive error token on failure so callers can degrade
        gracefully.
        """
        self._last_prompt   = json.dumps(messages)
        self._last_response = self.llm.chat(messages, max_tokens=max_tokens, temperature=temperature)
        return self._last_response

    # Keep _cf_call as an alias so any external callers aren't broken
    _cf_call = _llm_call

    # ══════════════════════════════════════════════════════════════════════════
    # CONTEXT BUILDER
    # ══════════════════════════════════════════════════════════════════════════

    def _build_context(self, run_id: str, max_artifact_chars: int = 300) -> str:
        """Compact state summary for LLM prompts."""
        aspects   = self.wm.get_goal_aspects(run_id)
        artifacts = self.wm.get_artifacts(run_id)

        lines: list[str] = []

        if aspects:
            lines.append("GOAL ASPECTS:")
            for a in aspects:
                lines.append(f"  [{a['status'].upper():12}] {a['aspect']}")

        tool_lines = self._tool_context_lines(run_id)
        if tool_lines:
            if lines:
                lines.append("")
            lines.extend(tool_lines)

        if artifacts:
            lines.append("\nRECENT ARTIFACTS (last 3):")
            for art in artifacts[-3:]:
                preview = (art.get("content") or "")[:max_artifact_chars]
                lines.append(f"  [{art['artifact_type']}] {art['title']}")
                if preview:
                    lines.append(f"    {preview[:200]}")

        return "\n".join(lines) if lines else "No prior work recorded."

    # ══════════════════════════════════════════════════════════════════════════
    # ACTION HANDLERS
    # ══════════════════════════════════════════════════════════════════════════

    def _handle_plan(
        self, run_id: str, goal: str, step_number: int
    ) -> dict[str, Any]:
        """
        Call CF LLM to decompose the goal into 2-4 trackable aspects.
        Saves aspects to working memory and a DOCUMENT artifact.
        """
        context = self._build_context(run_id)

        raw = self._llm_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a goal decomposition engine. "
                        "Output ONLY valid JSON. No prose. No markdown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal}\n\n"
                        f"CURRENT STATE:\n{context}\n\n"
                        "Decompose this goal into 3-5 high-level deliverable aspects.\n"
                        'Output ONLY: {"aspects": ["snake_case_name_1", ...]}\n'
                        "\nRules:\n"
                        "- MAXIMUM 5 aspects. Never exceed this.\n"
                        "- Each aspect = one major deliverable (research, code, document, delivery)\n"
                        "- Merge related sub-tasks — 'casualty_analysis' covers all analysis work,\n"
                        "  'python_report' covers writing AND running the script\n"
                        "- No meta-aspects: no 'script_testing', 'data_validation', 'review', "
                        "'formatting', 'compilation' — these are implicit steps, not deliverables\n"
                        "- No redundant aspects: 'report_generation' + 'data_formatting' = ONE aspect\n"
                        "- snake_case only, describe the deliverable not the action\n"
                        "- Specific to this goal, not generic\n"
                        "- CRITICAL: Only add 'email_transmission_confirmation' as the FINAL aspect "
                        "if the goal EXPLICITLY mentions sending or emailing someone (e.g. contains "
                        "an email address or the word 'email'/'send to'). "
                        "If no email delivery is requested, do NOT include it. "
                        "This is the ONLY delivery aspect — do NOT create any other aspect whose "
                        "name contains 'email', 'send', 'notify', 'deliver', 'draft', or 'transmission'."
                    ),
                },
            ],
            max_tokens  = 512,
            temperature = 0.2,
        )

        # Parse JSON -> fallback to regex -> fallback to sensible defaults
        print(f"[LiveExecutor] PLAN raw LLM response: {raw[:500]}", flush=True)
        aspects: list[str] = []
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                aspects = parsed.get("aspects", [])
                # Sanitize: force snake_case, strip whitespace
                aspects = [
                    re.sub(r"[^a-z0-9_]", "_", a.strip().lower().replace(" ", "_"))
                    for a in aspects if isinstance(a, str) and len(a.strip()) > 2
                ]
        except (json.JSONDecodeError, AttributeError) as exc:
            print(f"[LiveExecutor] PLAN JSON parse failed: {exc}", flush=True)

        # Hard cap — prompt says 5, code enforces 5
        aspects = aspects[:5]

        if not aspects:
            # Regex fallback — pull anything that looks like a snake_case identifier
            aspects = [
                t for t in re.findall(r'"([a-z][a-z0-9_]{2,40})"', raw)
                if "_" in t or len(t) > 6
            ][:5]

        if not aspects:
            # Last resort — LLM completely failed, use goal keywords to seed aspects
            print("[LiveExecutor] PLAN fallback: deriving aspects from goal text", flush=True)
            aspects = ["define_approach", "produce_deliverable", "verify_result"]

        # Persist aspects
        for asp in aspects:
            self.wm.upsert_goal_aspect(run_id, asp, "pending", step_number)

        # Persist plan artifact
        step_id = self.wm.record_step(
            run_id, step_number, "PLAN", "Goal Decomposition Plan", goal[:100]
        )
        plan_body = f"Goal: {goal}\n\nDecomposed aspects:\n" + \
                    "\n".join(f"- {a}" for a in aspects)
        self.wm.save_artifact(
            run_id, step_id, "DOCUMENT", "Plan: Goal Decomposition", plan_body
        )

        print(f"[LiveExecutor] PLAN -> aspects: {aspects}", flush=True)
        self._publish_narrative(run_id, f"Planned run into aspects: {', '.join(aspects)}")
        self._publish_aspects(run_id)
        return {
            "success"         : True,
            "exec_success"    : False,
            "exec_returncode" : -1,
            "aspects_created" : aspects,
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _handle_analyze(
        self, run_id: str, goal: str, step_number: int
    ) -> dict[str, Any]:
        """Structured analysis of current state -> ANALYSIS artifact."""
        context = self._build_context(run_id)

        content = self._llm_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a structured analysis engine. "
                        "Use tables and bullet lists only. No prose paragraphs."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal}\n\n"
                        f"CURRENT STATE:\n{context}\n\n"
                        "Provide a structured analysis of: what is established, "
                        "what gaps remain, and what the next highest-value action is."
                    ),
                },
            ],
            max_tokens  = 1024,
            temperature = 0.4,
        )

        step_id = self.wm.record_step(
            run_id, step_number, "ANALYZE", "Structured Analysis", goal[:100]
        )
        self.wm.save_artifact(
            run_id, step_id, "ANALYSIS", "Structured Analysis", content
        )
        self._publish_narrative(run_id, content[:1000])

        return {"success": True, "exec_success": False, "exec_returncode": -1}

    # ──────────────────────────────────────────────────────────────────────────

    def _handle_invoke_tool(
        self, run_id: str, goal: str, step_number: int
    ) -> dict[str, Any]:
        """
        LLM-routed tool invocation.

        1. Ask CF LLM which tool to call (from the capability manifest) and with
           what action / parameters.
        2. If manifest tools exist → invoke via webhook gateway.
           If no manifest tools (or LLM picks "web_search") → fall back to the
           built-in web search via core.web_tools.
        3. Synthesise the raw result with CF LLM → NARRATIVE artifact.
        """
        aspects = self.wm.get_goal_aspects(run_id)
        pending = [a["aspect"].replace("_", " ") for a in aspects if a["status"] == "pending"]
        context = self._build_context(run_id)

        # ── Step 1: LLM picks tool + params ───────────────────────────────────
        manifest    = self._get_capability_manifest(run_id)
        tool_list   = manifest.tools if manifest else []

        def _fmt_tool(t) -> str:
            schema    = t.input_schema or {}
            required  = schema.get("required", [])
            props     = schema.get("properties", {}) or {}
            param_strs = []
            for field_name in required:
                field_desc = (props.get(field_name) or {}).get("description", "")
                param_strs.append(f"{field_name}: {field_desc}" if field_desc else field_name)
            params_line = f"    required params: {', '.join(param_strs)}" if param_strs else ""
            return (
                f'  - name: "{t.name}"  actions: {t.actions}\n'
                f'    description: "{t.description[:240]}"\n'
                f'{params_line}'
            ).rstrip()

        tools_block = "\n".join(_fmt_tool(t) for t in tool_list) if tool_list else (
            '  - name: "web_search"  actions: ["search"]\n'
            '    description: "Search the web for information."\n'
            '    required params: query: Search query string'
        )

        print(
            f"[LiveExecutor] INVOKE_TOOL manifest tools: "
            f"{[t.name for t in tool_list] if tool_list else ['web_search (fallback)']}",
            flush=True,
        )

        # If ALL remaining pending aspects are delivery/tool-locked, add an
        # explicit routing directive so small LLMs don't default to web_search.
        pending_raw = [a["aspect"] for a in self.wm.get_goal_aspects(run_id) if a["status"] == "pending"]
        all_delivery = bool(pending_raw) and all(self._is_tool_locked(a) for a in pending_raw)
        delivery_directive = (
            "\nIMPORTANT: All remaining work is DELIVERY, not research. "
            "You MUST select the tool that sends or delivers output (e.g. send_email). "
            "Do NOT select web_search — the research phase is complete."
        ) if all_delivery else (
            "\nIf no delivery tool is needed, use web_search."
        )

        router_raw = self._llm_call(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a tool router. Output ONLY valid JSON. No prose.\n"
                        'Schema: {"tool": "<name>", "action": "<action>", "params": {...}}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal}\n\n"
                        f"PENDING ASPECTS: {', '.join(pending) or 'none'}\n\n"
                        f"AVAILABLE TOOLS:\n{tools_block}\n"
                        f"{delivery_directive}\n\n"
                        "Select the single best tool call. Output ONLY the JSON."
                    ),
                },
            ],
            max_tokens  = 256,
            temperature = 0.1,
        )

        # Parse the router's JSON
        tool_name   = "web_search"
        tool_action = "search"
        tool_params: dict[str, Any] = {}
        try:
            match = re.search(r"\{.*\}", router_raw, re.DOTALL)
            if match:
                parsed      = json.loads(match.group())
                tool_name   = str(parsed.get("tool",   "web_search")).strip()
                tool_action = str(parsed.get("action", "search")).strip()
                tool_params = parsed.get("params", {}) or {}
        except (json.JSONDecodeError, AttributeError):
            pass

        print(f"[LiveExecutor] INVOKE_TOOL → tool={tool_name} action={tool_action}", flush=True)

        # ── Email body composition ─────────────────────────────────────────────
        # When the router picks send_email, it only sees tool schemas — not the
        # research artifacts.  The resulting params.html is always a thin stub.
        # Do a dedicated LLM call here that reads ALL artifacts and composes a
        # proper HTML email body, then inject it into tool_params before sending.
        if tool_name == "send_email":
            body_html = str(tool_params.get("html", "") or tool_params.get("text", "") or "").strip()
            if len(body_html) < 200:   # stub or missing — compose from artifacts
                artifacts_all = self.wm.get_artifacts(run_id)

                # Separate code from prose artifacts so each gets the right formatting
                code_artifacts = [
                    a for a in artifacts_all if a.get("artifact_type") == "CODE"
                ]
                prose_artifacts = [
                    a for a in artifacts_all
                    if a.get("artifact_type") in ("DOCUMENT", "NARRATIVE", "ANALYSIS")
                ]

                prose_text = "\n\n".join(
                    f"[{a['artifact_type']}] {a['title']}\n{(a.get('content') or '')[:1500]}"
                    for a in prose_artifacts
                )[:5000]

                code_text = "\n\n".join(
                    f"# {a['title']}\n{(a.get('content') or '')[:3000]}"
                    for a in code_artifacts
                )[:4000]

                has_code = bool(code_text.strip())

                import html as _html
                system_prompt = (
                    "You are an expert technical writer. "
                    "Write a professional HTML email body covering the research findings. "
                    "Use <h2>, <p>, <ul>, <li>, <strong> tags only. "
                    "Do NOT include any code blocks — code will be appended separately. "
                    "Output ONLY the HTML — no subject line, no greeting, start with content."
                )
                user_content = (
                    f"GOAL: {goal}\n\n"
                    + (f"RESEARCH & ANALYSIS:\n{prose_text}\n\n" if prose_text else "")
                    + "Write a comprehensive HTML email summarising all findings and key data points."
                )

                composed = self._llm_call(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_content},
                    ],
                    max_tokens  = 2048,
                    temperature = 0.4,
                )

                # Programmatically append code blocks — never trust the LLM to
                # HTML-escape Python source.  Use html.escape() so <, >, &, " are
                # safe, and wrap in a styled <pre> that preserves whitespace.
                if has_code:
                    code_blocks_html = "<h2>Python Script</h2>"
                    for a in code_artifacts:
                        raw_code = (a.get("content") or "").strip()
                        # html.escape handles <, >, &, " — then convert newlines
                        # to <br> so every email client renders line breaks.
                        # white-space:pre is belt-and-suspenders for clients that
                        # support it; <br> is the universal fallback.
                        escaped = _html.escape(raw_code).replace("\n", "<br>\n")
                        code_blocks_html += (
                            f'<pre style="background:#f4f4f4;padding:16px;'
                            f'border-radius:6px;font-family:Courier New,monospace;'
                            f'font-size:13px;white-space:pre;overflow-x:auto;'
                            f'line-height:1.5;">'
                            f"<code>{escaped}</code></pre>"
                        )
                    composed = composed + code_blocks_html

                tool_params["html"] = composed
                tool_params.pop("body", None)   # remove any stub "body" key
                tool_params.pop("text", None)
                # Ensure recipient and subject have sensible defaults if missing
                if not tool_params.get("to"):
                    import re as _re
                    emails = _re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", goal)
                    tool_params["to"] = emails[0] if emails else "chris@xybernetex.com"
                if not tool_params.get("subject"):
                    tool_params["subject"] = f"Research Report: {goal[:80]}"
                # Pass per-request Resend key so self-hosted toolservers can use it
                if self._resend_api_key:
                    tool_params["resend_api_key"] = self._resend_api_key
                print("[LiveExecutor] INVOKE_TOOL email body composed from artifacts.", flush=True)

        # ── Step 2: Execute the tool ───────────────────────────────────────────
        raw_output: str = ""

        if tool_name == "web_search":
            # Built-in web search — always available, no gateway required
            from core.web_tools import search_web, format_search_results
            query = str(tool_params.get("query", "") or goal[:120])
            if not query:
                query = goal[:120]
            try:
                results    = search_web(
                    query,
                    num_results    = 3,
                    tavily_api_key = self._tavily_api_key,
                    brave_api_key  = self._brave_api_key,
                )
                raw_output = format_search_results(results)
            except Exception as exc:
                raw_output = f"Web search unavailable: {exc}"
        else:
            # Registered webhook tool
            step_id_for_tool = f"{run_id}_{step_number}"
            result = self._invoke_tool(
                run_id    = run_id,
                step_id   = step_id_for_tool,
                tool_name = tool_name,
                action    = tool_action,
                payload   = tool_params,
            )
            if result.succeeded:
                raw_output = str(result.output or "")
            else:
                raw_output = f"Tool error ({tool_name}.{tool_action}): {result.error}"
                print(f"[LiveExecutor] INVOKE_TOOL tool failure: {result.error}", flush=True)

        # ── Step 3: Synthesise result with CF LLM ─────────────────────────────
        content = self._llm_call(
            messages=[
                {
                    "role": "system",
                    "content": "Extract decision-relevant facts from tool output. Be concise.",
                },
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal}\n\n"
                        f"TOOL: {tool_name}.{tool_action}\n\n"
                        f"TOOL OUTPUT:\n{raw_output[:3000]}\n\n"
                        "Extract only facts directly relevant to the goal. Discard filler."
                    ),
                },
            ],
            max_tokens  = 768,
            temperature = 0.3,
        )

        step_id = self.wm.record_step(
            run_id, step_number, "INVOKE_TOOL",
            f"Tool: {tool_name}.{tool_action}", goal[:100]
        )
        self.wm.save_artifact(
            run_id, step_id, "NARRATIVE",
            f"Tool Result: {tool_name}.{tool_action}", content
        )
        self._publish_narrative(run_id, content[:1000])

        # ── Mark tool-locked aspects complete on successful delivery tool calls ──
        # web_search is informational — it never completes delivery aspects.
        # Any other tool (send_email, etc.) completing without error unlocks them.
        tool_succeeded = not raw_output.startswith("Tool error")
        if tool_name != "web_search" and tool_succeeded:
            aspects_now   = self.wm.get_goal_aspects(run_id)
            locked_pending = [
                a["aspect"] for a in aspects_now
                if a["status"] == "pending" and self._is_tool_locked(a["aspect"])
            ]
            if locked_pending:
                self.wm.upsert_goal_aspect(
                    run_id, locked_pending[0], "complete", step_number
                )
                self._publish_aspects(run_id)
                print(
                    f"[LiveExecutor] INVOKE_TOOL -> tool-locked aspect "
                    f"'{locked_pending[0]}' marked complete via {tool_name}",
                    flush=True,
                )

        return {"success": True, "exec_success": False, "exec_returncode": -1}

    # ──────────────────────────────────────────────────────────────────────────

    def _handle_write_artifact(
        self, run_id: str, goal: str, step_number: int
    ) -> dict[str, Any]:
        """
        LLM generates a deliverable — CODE or DOCUMENT depending on goal signals.
        CODE artifacts are written to disk for EXECUTE_CODE to consume.
        """
        context = self._build_context(run_id)
        aspects = self.wm.get_goal_aspects(run_id)
        pending = [a["aspect"] for a in aspects if a["status"] == "pending"]

        # Detect code vs document intent
        goal_tokens   = set(goal.lower().split())
        pending_str   = " ".join(pending)
        is_code       = bool(_CODE_SIGNALS & goal_tokens) and (
            any(s in pending_str.lower() for s in _CODE_SIGNALS)
            or any(s in pending_str.lower() for s in ("produce", "create", "write", "implement"))
            or not pending   # no aspects yet -> default to code for code goals
        )

        focus = (
            f"Produce executable Python code for: {goal}"
            if is_code
            else f"Produce the required deliverable for: {goal}"
        )
        if pending:
            focus += f"\n\nAddress specifically: {', '.join(pending[:2])}"

        if is_code:
            system = (
                "You are a Python code writer. "
                "Output ONLY raw Python code. No markdown fences. No explanation. "
                "Start with a comment describing what the script does. "
                "The code must be complete and immediately executable. "
                "CRITICAL: Use ONLY Python standard library modules "
                "(os, sys, json, csv, math, datetime, pathlib, collections, "
                "itertools, functools, re, argparse, etc.). "
                "Do NOT import pandas, numpy, requests, scipy, matplotlib, "
                "or any other third-party package — they are not available. "
                "Do NOT include any email sending, HTTP requests, or network "
                "calls in the script — output results to stdout only. "
                "A separate delivery tool handles sending; the script must only "
                "compute and print."
            )
        else:
            system = (
                "You are a precision information designer. "
                "Use tables and bullet lists only. No prose. No introduction. "
                "No conclusion. Start directly with the content."
            )

        # Inject full NARRATIVE content (web search results) so the LLM uses
        # real data instead of hallucinating.  Truncated previews in _build_context
        # are not enough — the full text is required for grounded output.
        all_artifacts = self.wm.get_artifacts(run_id)
        narratives = [
            a for a in all_artifacts if a.get("artifact_type") == "NARRATIVE"
        ]
        research_block = ""
        if narratives:
            parts = []
            for n in narratives:
                body = (n.get("content") or "").strip()
                if body:
                    parts.append(f"--- {n.get('title', 'Research')} ---\n{body}")
            if parts:
                research_block = (
                    "\n\nRESEARCH DATA (use this — do not invent numbers or facts):\n"
                    + "\n\n".join(parts)
                )

        content = self._llm_call(
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        f"GOAL: {goal}\n\n"
                        f"CONTEXT:\n{context}"
                        f"{research_block}\n\n"
                        f"TASK: {focus}"
                    ),
                },
            ],
            max_tokens  = 2048,
            temperature = 0.25 if is_code else 0.5,
        )

        artifact_type = "CODE" if is_code else "DOCUMENT"
        title         = f"{'Script' if is_code else 'Deliverable'}: {goal[:60]}"

        step_id = self.wm.record_step(
            run_id, step_number, "WRITE_ARTIFACT", title, focus[:120]
        )
        art_id = self.wm.save_artifact(
            run_id, step_id, artifact_type, title, content
        )

        # Write CODE to disk so EXECUTE_CODE can find it
        script_path: str | None = None
        if is_code:
            from core.code_executor import sanitize_code
            clean       = sanitize_code(content)
            script_path = str(
                self.output_dir / f"script_{run_id[:8]}_{step_number}.py"
            )
            Path(script_path).write_text(clean, encoding="utf-8")

            # Attach path to artifact metadata
            self.wm.conn.execute(
                "UPDATE artifacts SET metadata=? WHERE id=?",
                (json.dumps({"script_path": script_path}), art_id),
            )
            self.wm.conn.commit()
            print(f"[LiveExecutor] CODE artifact written -> {script_path}", flush=True)

        # Advance first pending aspect:
        # CODE artifacts -> "in_progress" (needs EXECUTE_CODE to complete)
        # DOCUMENT artifacts -> "complete" (the document IS the deliverable)
        # Tool-locked aspects are SKIPPED — only INVOKE_TOOL can complete those.
        if pending:
            advanceable = [p for p in pending if not self._is_tool_locked(p)]
            if advanceable:
                new_status = "in_progress" if is_code else "complete"
                self.wm.upsert_goal_aspect(
                    run_id, advanceable[0], new_status, step_number
                )
                self._publish_aspects(run_id)
            else:
                print(
                    "[LiveExecutor] WRITE_ARTIFACT -> all pending aspects are "
                    "tool-locked; skipping aspect advancement",
                    flush=True,
                )

        self._publish_narrative(
            run_id,
            f"{artifact_type} artifact generated: {title}",
        )

        return {
            "success"         : True,
            "exec_success"    : False,
            "exec_returncode" : -1,
            "artifact_id"     : art_id,
            "artifact_type"   : artifact_type,
            "script_path"     : script_path,
        }

    # ──────────────────────────────────────────────────────────────────────────

    def _handle_execute_code(
        self, run_id: str, goal: str, step_number: int
    ) -> dict[str, Any]:
        """
        Locate the most recent CODE artifact with a disk path, run it,
        persist the execution record, and mark aspects complete on success.
        """
        from core.code_executor import run_code

        # Find the most recent CODE artifact that has a script_path in metadata
        rows = self.wm.conn.execute(
            """
            SELECT id, title, metadata
            FROM   artifacts
            WHERE  run_id = ? AND artifact_type = 'CODE'
            ORDER  BY created_at DESC
            LIMIT  5
            """,
            (run_id,),
        ).fetchall()

        script_path: str | None = None
        art_id:      str | None = None

        for row in rows:
            meta = json.loads(row["metadata"] or "{}")
            sp   = meta.get("script_path")
            if sp and Path(sp).exists():
                script_path = sp
                art_id      = row["id"]
                break

        if not script_path:
            print("[LiveExecutor] EXECUTE_CODE -> no executable CODE artifact found", flush=True)
            return {
                "success"         : False,
                "exec_success"    : False,
                "exec_returncode" : -1,
                "error"           : "no_executable_code_artifact",
            }

        print(f"[LiveExecutor] EXECUTE_CODE -> {script_path}", flush=True)
        runner = self._execution_streamer if self._execution_streamer else run_code
        result = runner(script_path, timeout=30)

        # Persist execution record
        self.wm.save_execution(art_id, result)

        # Build narrative
        stdout    = (result.get("stdout") or "")[:1500]
        stderr    = (result.get("stderr") or "")[:500]
        exit_code = result.get("returncode", -1)
        status    = "SUCCESS" if result.get("success") else "FAILED"

        narrative_lines = [f"Exit code: {exit_code}  Status: {status}"]
        if stdout:
            narrative_lines += ["", "STDOUT:", stdout]
        if stderr:
            narrative_lines += ["", "STDERR:", stderr]
        narrative = "\n".join(narrative_lines)

        step_id = self.wm.record_step(
            run_id, step_number, "EXECUTE_CODE", f"Execute: {status}", script_path
        )
        self.wm.save_artifact(
            run_id, step_id, "NARRATIVE",
            f"Execution Result: {status}", narrative
        )
        self._publish_narrative(run_id, narrative)

        # On success -> mark all in_progress / pending aspects complete,
        # but leave tool-locked delivery aspects alone so INVOKE_TOOL fires them.
        if result.get("success"):
            for a in self.wm.get_goal_aspects(run_id):
                if a["status"] in ("pending", "in_progress") and not self._is_tool_locked(a["aspect"]):
                    self.wm.upsert_goal_aspect(
                        run_id, a["aspect"], "complete", step_number
                    )
            self._publish_aspects(run_id)

        print(
            f"[LiveExecutor] EXECUTE_CODE -> exit={exit_code} "
            f"stdout={len(stdout)}c stderr={len(stderr)}c",
            flush=True,
        )

        return {
            "success"         : True,
            "exec_success"    : result.get("success", False),
            "exec_returncode" : exit_code,
            "stdout"          : stdout,
            "stderr"          : stderr,
        }
