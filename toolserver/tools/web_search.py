"""
web_search.py — Tavily-powered web search tool

Invocation contract
-------------------
invocation["tool_name"]  = "web_search"
invocation["action"]     = "search"
invocation["input"]      = {
    "query":        str,           # required
    "max_results":  int,           # optional, default 5
    "search_depth": "basic"|"advanced",  # optional, default "basic"
    "include_answer": bool,        # optional, default True  (Tavily AI answer)
    "topic":        "general"|"news",    # optional, default "general"
}

Output
------
{
    "answer":  str,         # Tavily AI-synthesised answer (if available)
    "results": [            # raw search hits
        {
            "title":   str,
            "url":     str,
            "content": str,
            "score":   float,
        },
        ...
    ],
    "query": str,
}

Env vars
--------
TAVILY_API_KEY  — required; obtain from https://tavily.com
"""
from __future__ import annotations

import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("tool_server.tools.web_search")

_TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "").strip()
_TAVILY_ENDPOINT = "https://api.tavily.com/search"
_TIMEOUT = (5, 30)  # (connect, read) seconds


async def run(invocation: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a Tavily web search.

    Returns status="succeeded" with results, or status="failed" with an error.
    """
    if not _TAVILY_API_KEY:
        return {
            "status": "failed",
            "output": {},
            "error":  "TAVILY_API_KEY is not configured on the ToolServer",
        }

    params: dict[str, Any] = invocation.get("input", {}) or {}
    query: str = str(params.get("query", "")).strip()

    if not query:
        # Fall back to using the run_id goal if caller didn't pass a query
        query = str(invocation.get("run_id", "search")).strip()

    if not query:
        return {
            "status": "failed",
            "output": {},
            "error":  "web_search requires input.query",
        }

    payload = {
        "api_key":        _TAVILY_API_KEY,
        "query":          query,
        "max_results":    int(params.get("max_results", 5)),
        "search_depth":   str(params.get("search_depth", "basic")),
        "include_answer": bool(params.get("include_answer", True)),
        "topic":          str(params.get("topic", "general")),
    }

    logger.info("Tavily search: %r (depth=%s)", query, payload["search_depth"])

    try:
        response = requests.post(
            _TAVILY_ENDPOINT,
            json=payload,
            timeout=_TIMEOUT,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
    except requests.Timeout:
        return {
            "status": "timed_out",
            "output": {},
            "error":  "Tavily request timed out",
        }
    except requests.RequestException as exc:
        return {
            "status": "failed",
            "output": {},
            "error":  f"Tavily HTTP error: {exc}",
        }
    except ValueError as exc:
        return {
            "status": "failed",
            "output": {},
            "error":  f"Tavily invalid JSON response: {exc}",
        }

    # Normalise Tavily response into our output schema
    raw_results = data.get("results", [])
    results = [
        {
            "title":   str(r.get("title", "")),
            "url":     str(r.get("url", "")),
            "content": str(r.get("content", ""))[:2000],
            "score":   float(r.get("score", 0.0)),
        }
        for r in raw_results
        if isinstance(r, dict)
    ]

    output = {
        "answer":  str(data.get("answer", "")),
        "results": results,
        "query":   query,
    }

    logger.info(
        "Tavily returned %d results for %r", len(results), query
    )

    return {
        "status": "succeeded",
        "output": output,
        "error":  "",
    }
