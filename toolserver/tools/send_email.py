"""
send_email.py — Resend-powered email tool

Invocation contract
-------------------
invocation["tool_name"]  = "send_email"
invocation["action"]     = "send"
invocation["input"]      = {
    "to":        str | list[str],   # required — one address or list
    "subject":   str,               # required
    "html":      str,               # required (or "text")
    "text":      str,               # optional plain-text fallback
    "from":      str,               # optional — defaults to RESEND_FROM_ADDRESS
    "cc":        str | list[str],   # optional
    "bcc":       str | list[str],   # optional
    "reply_to":  str | list[str],   # optional
    "tags":      list[{"name": str, "value": str}],  # optional Resend tags
}

Output (on success)
------
{
    "id":      str,   # Resend message ID
    "to":      list,
    "subject": str,
}

Env vars
--------
RESEND_API_KEY      — required; obtain from https://resend.com/api-keys
RESEND_FROM_ADDRESS — optional default sender, e.g. "Xybernetex <noreply@xybernetex.com>"
                      Can be overridden per-invocation via input.from
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tool_server.tools.send_email")

_RESEND_API_KEY: str    = os.getenv("RESEND_API_KEY", "").strip()
_RESEND_FROM: str       = os.getenv("RESEND_FROM_ADDRESS", "").strip()
_RESEND_ENDPOINT        = "https://api.resend.com/emails"
_TIMEOUT                = (5, 30)  # (connect, read) seconds


def _to_list(value: Any) -> list[str]:
    """Normalise a str-or-list address field into a list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]


async def run(invocation: dict[str, Any]) -> dict[str, Any]:
    """
    Send an email via the Resend API.

    Returns status="succeeded" with the Resend message ID,
    or status="failed" with a descriptive error.
    """
    if not _RESEND_API_KEY:
        return {
            "status": "failed",
            "output": {},
            "error":  "RESEND_API_KEY is not configured on the ToolServer",
        }

    params: dict[str, Any] = invocation.get("input", {}) or {}

    # ── Required fields ───────────────────────────────────────────────────────
    to: list[str] = _to_list(params.get("to"))
    subject: str  = str(params.get("subject", "")).strip()

    if not to:
        return {
            "status": "failed",
            "output": {},
            "error":  "send_email requires input.to (recipient address)",
        }
    if not subject:
        return {
            "status": "failed",
            "output": {},
            "error":  "send_email requires input.subject",
        }

    html: str = str(params.get("html", "")).strip()
    text: str = str(params.get("text", "")).strip()

    if not html and not text:
        return {
            "status": "failed",
            "output": {},
            "error":  "send_email requires input.html or input.text (email body)",
        }

    # ── Sender ────────────────────────────────────────────────────────────────
    from_address: str = str(params.get("from", "") or _RESEND_FROM).strip()
    if not from_address:
        return {
            "status": "failed",
            "output": {},
            "error":  (
                "send_email requires a sender address. "
                "Pass input.from or set RESEND_FROM_ADDRESS in the ToolServer .env"
            ),
        }

    # ── Build payload ─────────────────────────────────────────────────────────
    payload: dict[str, Any] = {
        "from":    from_address,
        "to":      to,
        "subject": subject,
    }

    if html:
        payload["html"] = html
    if text:
        payload["text"] = text

    cc = _to_list(params.get("cc"))
    if cc:
        payload["cc"] = cc

    bcc = _to_list(params.get("bcc"))
    if bcc:
        payload["bcc"] = bcc

    reply_to = _to_list(params.get("reply_to"))
    if reply_to:
        payload["reply_to"] = reply_to

    tags = params.get("tags")
    if isinstance(tags, list) and tags:
        payload["tags"] = tags

    logger.info(
        "Resend send: to=%s  subject=%r  from=%s",
        to, subject, from_address,
    )

    # ── Call Resend API ───────────────────────────────────────────────────────
    try:
        response = requests.post(
            _RESEND_ENDPOINT,
            json=payload,
            headers={
                "Authorization": f"Bearer {_RESEND_API_KEY}",
                "Content-Type":  "application/json",
            },
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()

    except requests.Timeout:
        return {
            "status": "timed_out",
            "output": {},
            "error":  "Resend request timed out",
        }
    except requests.HTTPError as exc:
        # Resend returns error detail in the body
        try:
            err_body = exc.response.json()
            err_msg  = err_body.get("message") or err_body.get("name") or str(exc)
        except Exception:
            err_msg = str(exc)
        return {
            "status": "failed",
            "output": {},
            "error":  f"Resend API error ({exc.response.status_code}): {err_msg}",
        }
    except requests.RequestException as exc:
        return {
            "status": "failed",
            "output": {},
            "error":  f"Resend HTTP error: {exc}",
        }
    except ValueError as exc:
        return {
            "status": "failed",
            "output": {},
            "error":  f"Resend invalid JSON response: {exc}",
        }

    message_id: str = str(data.get("id", ""))
    logger.info("Resend accepted message id=%s to=%s", message_id, to)

    return {
        "status": "succeeded",
        "output": {
            "id":      message_id,
            "to":      to,
            "subject": subject,
        },
        "error": "",
    }
