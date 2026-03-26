"""
api/main.py — Xybernetex OpenEngine REST API

Run locally:
    cd OpenEngine
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /v1/run        Run a goal and return structured results
    GET  /v1/health     Liveness check
    GET  /v1/providers  List supported LLM providers and their default models
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is on sys.path
_API_DIR   = Path(__file__).resolve().parent
_REPO_ROOT = _API_DIR.parent
for _p in (_REPO_ROOT, _REPO_ROOT / "engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env", override=False)

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sdk import Engine, GoalResult

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "Xybernetex OpenEngine API",
    description = "RL-powered autonomous agent — bring your own LLM key.",
    version     = "0.1.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal    : str = Field(..., description="Plain-English goal for the agent to complete.")
    aspects : list[str] | None = Field(
        None,
        description=(
            "Optional pre-defined aspect list — skips PLAN and injects these directly. "
            "Last item should be 'email_transmission_confirmation' if email delivery is needed."
        ),
    )
    run_id  : str = Field("", description="Optional explicit run ID. Auto-generated if omitted.")

    # LLM credentials — all optional, fall back to server .env if omitted
    llm_provider   : str = Field("", description="openai | anthropic | gemini | mistral | cloudflare")
    llm_api_key    : str = Field("", description="API key for OpenAI / Anthropic / Gemini / Mistral.")
    llm_model      : str = Field("", description="Model ID override (e.g. 'gpt-4o', 'claude-opus-4-6').")
    cf_account_id  : str = Field("", description="Cloudflare account ID (cloudflare provider only).")
    cf_api_token   : str = Field("", description="Cloudflare API token (cloudflare provider only).")

    # Tool credentials — all optional, fall back to server .env if omitted
    tavily_api_key : str = Field("", description="Tavily API key for web search (tavily.com).")
    brave_api_key  : str = Field("", description="Brave Search API key (alternative to Tavily).")
    resend_api_key : str = Field("", description="Resend API key for email delivery (resend.com).")

    class Config:
        json_schema_extra = {
            "example": {
                "goal"         : "Research the top EV battery suppliers and email a report to me@example.com",
                "llm_provider" : "openai",
                "llm_api_key"  : "sk-...",
            }
        }


class ArtifactOut(BaseModel):
    type    : str
    title   : str
    content : str
    path    : str = ""


class AspectOut(BaseModel):
    name    : str
    status  : str
    complete: bool


class RunResponse(BaseModel):
    run_id   : str
    status   : str
    success  : bool
    steps    : int
    reward   : float
    goal     : str
    artifacts: list[ArtifactOut]
    aspects  : list[AspectOut]
    log_path : str
    elapsed_seconds: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}


@app.get("/v1/providers")
def providers() -> dict[str, Any]:
    return {
        "providers": [
            {
                "id"           : "openai",
                "name"         : "OpenAI",
                "default_model": "gpt-4o",
                "key_env_var"  : "LLM_API_KEY",
            },
            {
                "id"           : "anthropic",
                "name"         : "Anthropic",
                "default_model": "claude-opus-4-6",
                "key_env_var"  : "LLM_API_KEY",
            },
            {
                "id"           : "gemini",
                "name"         : "Google Gemini",
                "default_model": "gemini-2.0-flash",
                "key_env_var"  : "LLM_API_KEY",
            },
            {
                "id"           : "mistral",
                "name"         : "Mistral AI",
                "default_model": "mistral-large-latest",
                "key_env_var"  : "LLM_API_KEY",
            },
            {
                "id"           : "cloudflare",
                "name"         : "Cloudflare Workers AI",
                "default_model": "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                "key_env_var"  : "CF_API_TOKEN",
                "note"         : "Also requires CF_ACCOUNT_ID.",
            },
        ]
    }


@app.post("/v1/run", response_model=RunResponse)
def run_goal(req: RunRequest) -> RunResponse:
    """
    Run a goal to completion.

    Pass your LLM credentials in the request body — the server never stores them.
    If no credentials are provided, the server falls back to its own .env configuration.
    """
    t0 = time.perf_counter()

    try:
        engine = Engine(
            llm_provider             = req.llm_provider,
            llm_api_key              = req.llm_api_key,
            llm_model                = req.llm_model,
            cf_account_id            = req.cf_account_id,
            cf_api_token             = req.cf_api_token,
            tavily_api_key           = req.tavily_api_key,
            brave_api_key            = req.brave_api_key,
            resend_api_key           = req.resend_api_key,
        )
        result: GoalResult = engine.run(
            goal    = req.goal,
            aspects = req.aspects,
            run_id  = req.run_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0

    return RunResponse(
        run_id   = result.run_id,
        status   = result.status,
        success  = result.success,
        steps    = result.steps,
        reward   = result.reward,
        goal     = result.goal,
        elapsed_seconds = round(elapsed, 2),
        log_path = result.log_path,
        artifacts= [
            ArtifactOut(
                type    = a.type,
                title   = a.title,
                content = a.content,
                path    = a.path,
            )
            for a in result.artifacts
        ],
        aspects  = [
            AspectOut(name=a.name, status=a.status, complete=a.complete)
            for a in result.aspects
        ],
    )
