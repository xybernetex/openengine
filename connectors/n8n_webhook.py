"""
n8n_webhook.py - n8n-backed implementation of the stable ToolGateway contract.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from typing import Any

import requests

from connectors.contracts import ToolGateway, ToolInvocation, ToolResult


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class N8NWebhookConnector(ToolGateway):
    """
    Executes stable tool invocations against a dedicated n8n webhook endpoint.

    The worker never sees n8n-specific workflow details. It sends one canonical
    envelope and expects one canonical result payload back.
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        signing_secret: str | None = None,
        connect_timeout: float | None = None,
        read_timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.webhook_url = webhook_url or os.getenv("XYBER_TOOL_WEBHOOK_URL", "")
        self.signing_secret = signing_secret or os.getenv("XYBER_TOOL_WEBHOOK_SECRET", "")
        self.connect_timeout = float(
            connect_timeout if connect_timeout is not None
            else os.getenv("XYBER_TOOL_CONNECT_TIMEOUT", "5")
        )
        self.read_timeout = float(
            read_timeout if read_timeout is not None
            else os.getenv("XYBER_TOOL_READ_TIMEOUT", "60")
        )
        self.session = session or requests.Session()

    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        if not self.webhook_url:
            return ToolResult.failure(
                invocation,
                "connector_unconfigured: XYBER_TOOL_WEBHOOK_URL is not set",
            )

        payload = invocation.to_dict()
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        headers = {
            "Content-Type": "application/json",
            "X-Xybernetex-Contract-Version": invocation.contract_version,
            "X-Xybernetex-Invocation-Id": invocation.invocation_id,
        }

        if self.signing_secret:
            signature = hmac.new(
                self.signing_secret.encode("utf-8"),
                body.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            headers["X-Xybernetex-Signature"] = signature

        try:
            response = self.session.post(
                self.webhook_url,
                data=body,
                headers=headers,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            response.raise_for_status()
        except requests.Timeout:
            return ToolResult.failure(invocation, "connector_timeout", status="timed_out")
        except requests.RequestException as exc:
            return ToolResult.failure(invocation, f"connector_http_error: {exc}")

        try:
            raw = response.json()
        except ValueError as exc:
            return ToolResult.failure(invocation, f"connector_invalid_json: {exc}")

        if not isinstance(raw, dict):
            return ToolResult.failure(invocation, "connector_invalid_payload: expected object")

        return self._normalize_result(invocation, raw)

    def _normalize_result(
        self,
        invocation: ToolInvocation,
        payload: dict[str, Any],
    ) -> ToolResult:
        status = str(payload.get("status") or "failed")
        if status not in {"succeeded", "failed", "timed_out"}:
            status = "failed"
        return ToolResult(
            invocation_id=str(payload.get("invocation_id") or invocation.invocation_id),
            run_id=str(payload.get("run_id") or invocation.run_id),
            status=status,
            output=payload.get("output") if isinstance(payload.get("output"), dict) else {},
            error=str(payload.get("error") or ""),
            completed_at=str(payload.get("completed_at") or _utc_now()),
        )
