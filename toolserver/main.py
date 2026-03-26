"""
main.py — Xybernetex ToolServer

Lightweight FastAPI webhook listener that receives ToolInvocation requests
from the Worker and routes them to registered tool handlers.

This service is the N8N-independent execution layer. It speaks the same
stable ToolInvocation / ToolResult wire contract that N8NWebhookConnector
already uses, so the Worker can point XYBER_TOOL_WEBHOOK_URL at either
N8N or this server with zero code changes.

Auth
----
All requests must carry a valid HMAC-SHA256 signature in the
X-Xybernetex-Signature header. The secret is XYBER_TOOL_WEBHOOK_SECRET.
Requests without a valid signature are rejected with 401.

Request / Response Contract
----------------------------
POST /invoke
  Body  : ToolInvocation JSON  (same payload N8NWebhookConnector sends)
  Return: ToolResult JSON       (same payload N8NWebhookConnector normalises)

GET  /health
  Return: {"status": "ok", "tools": [...registered tool names...]}

Plugin System
-------------
Each file in tools/ that does NOT start with _ is a tool module.
It must export an async function:

    async def run(invocation: dict) -> dict

The returned dict must contain at minimum:
    {"status": "succeeded"|"failed", "output": {...}, "error": "..."}

The registry auto-discovers tools at startup — no registration code needed.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timezone

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from registry import ToolRegistry

# ── Env ───────────────────────────────────────────────────────────────────────
load_dotenv()

_SIGNING_SECRET: str = os.getenv("XYBER_TOOL_WEBHOOK_SECRET", "").strip()
_HOST: str           = os.getenv("TOOL_SERVER_HOST", "0.0.0.0")
_PORT: int           = int(os.getenv("TOOL_SERVER_PORT", "9000"))
_LOG_LEVEL: str      = os.getenv("LOG_LEVEL", "info").lower()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tool_server")

if not _SIGNING_SECRET:
    logger.warning(
        "XYBER_TOOL_WEBHOOK_SECRET is not set. "
        "All requests will be accepted WITHOUT signature verification. "
        "Set this in .env before deploying to production."
    )

# ── App + registry ────────────────────────────────────────────────────────────
app      = FastAPI(title="Xybernetex ToolServer", version="1.0.0")
registry = ToolRegistry()


@app.on_event("startup")
async def _on_startup() -> None:
    registry.discover()
    logger.info("ToolServer started. Registered tools: %s", registry.tool_names())


# ── Auth middleware ────────────────────────────────────────────────────────────

def _verify_signature(body: bytes, header_sig: str) -> bool:
    """Return True if the HMAC-SHA256 of body matches header_sig."""
    expected = hmac.new(
        _SIGNING_SECRET.encode(),
        body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, header_sig.strip())


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "tools":  registry.tool_names(),
        "time":   datetime.now(timezone.utc).isoformat(),
    })


@app.post("/invoke")
async def invoke(request: Request) -> JSONResponse:
    body = await request.body()

    # Signature check (skip if no secret configured)
    if _SIGNING_SECRET:
        sig = request.headers.get("X-Xybernetex-Signature", "")
        if not sig:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing X-Xybernetex-Signature header",
            )
        if not _verify_signature(body, sig):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid signature",
            )

    # Parse body
    try:
        import json
        invocation: dict = json.loads(body)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON body: {exc}",
        )

    tool_name: str = str(invocation.get("tool_name", "")).strip()
    if not tool_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invocation.tool_name is required",
        )

    logger.info(
        "INVOKE  tool=%s  action=%s  run=%s",
        tool_name,
        invocation.get("action", ""),
        invocation.get("run_id", ""),
    )

    # Dispatch to tool handler
    result = await registry.invoke(invocation)

    logger.info(
        "RESULT  tool=%s  status=%s  run=%s",
        tool_name,
        result.get("status"),
        invocation.get("run_id", ""),
    )

    return JSONResponse(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=_HOST,
        port=_PORT,
        log_level=_LOG_LEVEL,
        reload=False,
    )
