"""
api/main.py — Xybernetex OpenEngine REST API

Run locally:
    cd OpenEngine
    uvicorn api.main:app --reload --port 8000

Endpoints:
    POST /v1/run        Run a goal and return structured results
    GET  /v1/health     Liveness check
    GET  /v1/providers  List supported LLM providers and their default models
    GET  /v1/stats      Aggregated run analytics (JSON)
    GET  /stats         Simple HTML analytics dashboard
"""
from __future__ import annotations

import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
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

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from sdk import Engine, GoalResult

# ── Analytics DB ──────────────────────────────────────────────────────────────

_db_env = (
    os.getenv("DATABASE_PATH", "").strip()
    or os.getenv("DATABASE_URL", "").strip().removeprefix("sqlite:///")
)
if _db_env:
    _ANALYTICS_DB = Path(_db_env).parent / "analytics.db"
else:
    _ANALYTICS_DB = _REPO_ROOT / "core" / "memory" / "analytics.db"

_ANALYTICS_DB.parent.mkdir(parents=True, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_ANALYTICS_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT    NOT NULL,
            run_id          TEXT    NOT NULL,
            llm_provider    TEXT    NOT NULL DEFAULT '',
            status          TEXT    NOT NULL DEFAULT '',
            success         INTEGER NOT NULL DEFAULT 0,
            steps           INTEGER NOT NULL DEFAULT 0,
            reward          REAL    NOT NULL DEFAULT 0.0,
            elapsed_seconds REAL    NOT NULL DEFAULT 0.0,
            goal_len        INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def _record_run(
    run_id: str,
    llm_provider: str,
    status: str,
    success: bool,
    steps: int,
    reward: float,
    elapsed: float,
    goal_len: int,
) -> None:
    try:
        conn = _get_db()
        conn.execute(
            """INSERT INTO api_runs
               (ts, run_id, llm_provider, status, success, steps, reward, elapsed_seconds, goal_len)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                run_id,
                llm_provider or "unknown",
                status,
                1 if success else 0,
                steps,
                reward,
                round(elapsed, 2),
                goal_len,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # analytics must never break the main request


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


@app.get("/v1/stats")
def stats() -> dict[str, Any]:
    """Aggregated run analytics."""
    try:
        conn = _get_db()
        cur  = conn.cursor()

        total      = cur.execute("SELECT COUNT(*) FROM api_runs").fetchone()[0]
        today      = cur.execute("SELECT COUNT(*) FROM api_runs WHERE ts >= date('now')").fetchone()[0]
        this_week  = cur.execute("SELECT COUNT(*) FROM api_runs WHERE ts >= date('now', '-7 days')").fetchone()[0]
        this_month = cur.execute("SELECT COUNT(*) FROM api_runs WHERE ts >= date('now', '-30 days')").fetchone()[0]
        succeeded  = cur.execute("SELECT COUNT(*) FROM api_runs WHERE success = 1").fetchone()[0]
        success_rate = round(succeeded / total * 100, 1) if total else 0.0

        avg_steps   = cur.execute("SELECT AVG(steps) FROM api_runs").fetchone()[0] or 0
        avg_elapsed = cur.execute("SELECT AVG(elapsed_seconds) FROM api_runs").fetchone()[0] or 0
        avg_reward  = cur.execute("SELECT AVG(reward) FROM api_runs").fetchone()[0] or 0

        providers_raw = cur.execute(
            "SELECT llm_provider, COUNT(*) as cnt FROM api_runs GROUP BY llm_provider ORDER BY cnt DESC"
        ).fetchall()
        providers_breakdown = {row["llm_provider"]: row["cnt"] for row in providers_raw}

        recent = cur.execute(
            "SELECT ts, run_id, llm_provider, status, success, steps, reward, elapsed_seconds "
            "FROM api_runs ORDER BY id DESC LIMIT 10"
        ).fetchall()
        recent_runs = [dict(r) for r in recent]

        conn.close()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {
        "total_runs"          : total,
        "runs_today"          : today,
        "runs_this_week"      : this_week,
        "runs_this_month"     : this_month,
        "success_rate_pct"    : success_rate,
        "avg_steps"           : round(avg_steps, 1),
        "avg_elapsed_seconds" : round(avg_elapsed, 1),
        "avg_reward"          : round(avg_reward, 2),
        "providers"           : providers_breakdown,
        "recent_runs"         : recent_runs,
    }


@app.get("/stats", response_class=HTMLResponse)
def stats_dashboard() -> str:
    """Simple HTML analytics dashboard."""
    s = stats()
    provider_rows = "".join(
        f"<tr><td>{p}</td><td>{c}</td></tr>"
        for p, c in s["providers"].items()
    )
    recent_rows = "".join(
        f"<tr>"
        f"<td>{r['ts'][:19].replace('T',' ')}</td>"
        f"<td style='font-size:11px'>{r['run_id']}</td>"
        f"<td>{r['llm_provider']}</td>"
        f"<td>{'✅' if r['success'] else '❌'}</td>"
        f"<td>{r['steps']}</td>"
        f"<td>{r['elapsed_seconds']}s</td>"
        f"</tr>"
        for r in s["recent_runs"]
    )
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OpenEngine Stats</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f0f0f; color: #e0e0e0; margin: 0; padding: 32px; }}
  h1   {{ font-size: 24px; color: #fff; margin-bottom: 24px; }}
  h2   {{ font-size: 16px; color: #aaa; margin: 32px 0 12px; text-transform: uppercase; letter-spacing: 1px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; }}
  .card  {{ background: #1a1a1a; border: 1px solid #2a2a2a; border-radius: 8px;
            padding: 20px 28px; min-width: 140px; }}
  .card .val {{ font-size: 36px; font-weight: 700; color: #fff; }}
  .card .lbl {{ font-size: 12px; color: #888; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; background: #1a1a1a;
           border: 1px solid #2a2a2a; border-radius: 8px; overflow: hidden; }}
  th    {{ background: #222; color: #888; font-size: 11px; text-transform: uppercase;
           letter-spacing: 1px; padding: 10px 14px; text-align: left; }}
  td    {{ padding: 10px 14px; border-top: 1px solid #222; font-size: 13px; }}
  tr:hover td {{ background: #222; }}
</style>
</head>
<body>
<h1>⚡ OpenEngine — Live Stats</h1>
<div class="cards">
  <div class="card"><div class="val">{s['total_runs']}</div><div class="lbl">Total Runs</div></div>
  <div class="card"><div class="val">{s['runs_today']}</div><div class="lbl">Today</div></div>
  <div class="card"><div class="val">{s['runs_this_week']}</div><div class="lbl">This Week</div></div>
  <div class="card"><div class="val">{s['success_rate_pct']}%</div><div class="lbl">Success Rate</div></div>
  <div class="card"><div class="val">{s['avg_steps']}</div><div class="lbl">Avg Steps</div></div>
  <div class="card"><div class="val">{s['avg_elapsed_seconds']}s</div><div class="lbl">Avg Duration</div></div>
</div>

<h2>Providers</h2>
<table><tr><th>Provider</th><th>Runs</th></tr>{provider_rows}</table>

<h2>Recent Runs</h2>
<table>
  <tr><th>Time</th><th>Run ID</th><th>Provider</th><th>OK</th><th>Steps</th><th>Duration</th></tr>
  {recent_rows}
</table>
</body>
</html>"""


@app.post("/v1/run", response_model=RunResponse)
def run_goal(req: RunRequest, request: Request) -> RunResponse:
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

    _record_run(
        run_id       = result.run_id,
        llm_provider = req.llm_provider or "server_default",
        status       = result.status,
        success      = result.success,
        steps        = result.steps,
        reward       = result.reward,
        elapsed      = elapsed,
        goal_len     = len(req.goal),
    )

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
