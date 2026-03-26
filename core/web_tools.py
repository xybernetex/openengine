"""
web_tools.py — Web access for the Xybernetex agent.

Three modes (priority order):
  1. Tavily mode  — set TAVILY_API_KEY in .env.  Best quality: returns extracted
                    page content with search results, saving FETCH_URL steps.
  2. Brave mode   — set BRAVE_API_KEY in .env.  Direct Brave Search API.
                    fetch_url uses urllib for page extraction.
  3. n8n mode     — set N8N_BASE_URL in .env.  Routes through your n8n instance
                    over Tailscale (legacy behaviour, fully preserved).

Config (.env):
    # Tavily mode (recommended — includes content extraction)
    TAVILY_API_KEY=tvly-...       # get key at https://tavily.com

    # Brave mode (fallback)
    BRAVE_API_KEY=BSA...          # get free key at https://brave.com/search/api/

    # n8n mode (legacy / optional)
    N8N_BASE_URL=http://<tailscale-hostname>:5678
    N8N_SEARCH_PATH=xybernetex-search
    N8N_FETCH_PATH=xybernetex-fetch
    N8N_WEBHOOK_SECRET=
"""

import json
import os
import re
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional

# Load .env before reading env vars — web_tools.py is imported before cf_agent.py
# calls load_dotenv(), so without this the module-level os.getenv() calls always
# return "" even when keys are set in the .env file.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(Path(__file__).parent.parent / ".env", override=False)
    # Also try the Worker directory itself in case .env lives there
    _load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass   # python-dotenv not installed — env vars must be set externally

# ── Config ─────────────────────────────────────────────────────────────────────
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY", "")
TAVILY_API_URL  = "https://api.tavily.com/search"

BRAVE_API_KEY   = os.getenv("BRAVE_API_KEY", "")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

N8N_BASE_URL    = os.getenv("N8N_BASE_URL", "").rstrip("/")
SEARCH_PATH     = os.getenv("N8N_SEARCH_PATH", "xybernetex-search")
FETCH_PATH      = os.getenv("N8N_FETCH_PATH",  "xybernetex-fetch")
WEBHOOK_SECRET  = os.getenv("N8N_WEBHOOK_SECRET", "")

REQUEST_TIMEOUT = 30   # seconds


class WebToolsError(Exception):
    """Raised when search or fetch fails."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# TAVILY MODE — search with extracted content
# ══════════════════════════════════════════════════════════════════════════════

def _tavily_search(query: str, num_results: int = 8,
                   search_depth: str = "basic", api_key: str = "") -> dict:
    """
    Call Tavily Search API.

    Returns the same shape as _brave_search() for drop-in compatibility,
    but each result also includes 'content' (extracted page text) when
    available — saving the agent a FETCH_URL step.

    Args:
        query:        Search query string.
        num_results:  Max results to return (1-20).
        search_depth: "basic" (fast, cheap) or "advanced" (deeper crawl).
        api_key:      Override TAVILY_API_KEY env var for per-request keys.
    """
    key = api_key or TAVILY_API_KEY
    payload = json.dumps({
        "query": query,
        "max_results": min(num_results, 20),
        "search_depth": search_depth,
        "include_answer": True,
        "include_raw_content": False,
    }).encode()

    req = urllib.request.Request(TAVILY_API_URL, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise WebToolsError(f"Tavily API returned HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise WebToolsError(f"Cannot reach Tavily API: {e}")
    except json.JSONDecodeError as e:
        raise WebToolsError(f"Tavily returned non-JSON: {e}")

    results = []
    for item in data.get("results", [])[:num_results]:
        # Tavily returns published_date as ISO string; convert to human-readable age
        pub_date = item.get("published_date", "")
        age = _iso_to_age(pub_date) if pub_date else ""

        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("content", "")[:500],   # Tavily's 'content' is the snippet
            "age":     age,
            "source":  _hostname(item.get("url", "")),
            # Tavily bonus: extracted page content (not available from Brave)
            "raw_content": item.get("raw_content", ""),
        })

    return {
        "query":        query,
        "total_found":  len(data.get("results", [])),
        "results":      results,
        # Tavily bonus: AI-generated answer across all results
        "answer":       data.get("answer", ""),
    }


def _tavily_extract(url: str, max_chars: int = 8000, api_key: str = "") -> dict:
    """
    Use Tavily Extract API to get clean page content.
    Falls back to _direct_fetch() if the extract endpoint fails.
    """
    key = api_key or TAVILY_API_KEY
    extract_url = "https://api.tavily.com/extract"
    payload = json.dumps({
        "urls": [url],
    }).encode()

    req = urllib.request.Request(extract_url, data=payload, headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        # Tavily extract failed — fall back to direct fetch
        return _direct_fetch(url, max_chars)

    results = data.get("results", [])
    if not results:
        return _direct_fetch(url, max_chars)

    page = results[0]
    content = page.get("raw_content", "") or ""
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return {
        "url":         url,
        "title":       page.get("title", "") or _hostname(url),
        "description": "",
        "content":     content,
        "length":      len(content),
        "truncated":   truncated,
    }


def _hostname(url: str) -> str:
    """Extract hostname from a URL."""
    try:
        return urllib.parse.urlparse(url).hostname or ""
    except Exception:
        return ""


def _iso_to_age(iso_date: str) -> str:
    """Convert ISO date string to human-readable age like '3 days ago'."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta = now - dt
        days = delta.days
        if days == 0:
            hours = delta.seconds // 3600
            if hours == 0:
                return "just now"
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        if days < 30:
            return f"{days} day{'s' if days != 1 else ''} ago"
        months = days // 30
        if months < 12:
            return f"{months} month{'s' if months != 1 else ''} ago"
        years = months // 12
        return f"{years} year{'s' if years != 1 else ''} ago"
    except Exception:
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# BRAVE MODE — Direct Brave Search API (fallback)
# ══════════════════════════════════════════════════════════════════════════════

def _brave_search(query: str, num_results: int = 8, api_key: str = "") -> dict:
    """Call Brave Search API directly."""
    key = api_key or BRAVE_API_KEY
    params = f"?q={urllib.parse.quote(query)}&count={min(num_results, 20)}&safesearch=moderate"
    url = BRAVE_SEARCH_URL + params

    req = urllib.request.Request(url, headers={
        "Accept":               "application/json",
        "Accept-Encoding":      "gzip",
        "X-Subscription-Token": key,
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read()
            # Handle gzip encoding
            if resp.info().get("Content-Encoding") == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise WebToolsError(f"Brave Search API returned HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise WebToolsError(f"Cannot reach Brave Search API: {e}")
    except json.JSONDecodeError as e:
        raise WebToolsError(f"Brave Search returned non-JSON: {e}")

    results = []
    for item in data.get("web", {}).get("results", [])[:num_results]:
        results.append({
            "title":   item.get("title", ""),
            "url":     item.get("url", ""),
            "snippet": item.get("description", ""),
            "age":     item.get("age", ""),
            "source":  item.get("meta_url", {}).get("hostname", ""),
        })

    return {
        "query":       query,
        "total_found": data.get("web", {}).get("totalCount", len(results)),
        "results":     results,
    }


def _strip_html(html: str) -> str:
    """Basic HTML → plain text: remove tags, decode common entities."""
    # Remove scripts and styles entirely
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with newlines
    html = re.sub(r"<(p|div|br|li|tr|h[1-6])[^>]*>", "\n", html, flags=re.IGNORECASE)
    # Strip remaining tags
    html = re.sub(r"<[^>]+>", " ", html)
    # Decode common HTML entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
        "&apos;": "'", "&nbsp;": " ", "&#39;": "'", "&#x27;": "'",
    }
    for ent, char in entities.items():
        html = html.replace(ent, char)
    # Collapse whitespace
    html = re.sub(r"[ \t]+", " ", html)
    html = re.sub(r"\n{3,}", "\n\n", html)
    return html.strip()


def _direct_fetch(url: str, max_chars: int = 8000) -> dict:
    """Fetch a URL and extract plain text directly (no external API required)."""
    req = urllib.request.Request(url, headers={
        "User-Agent": (
            "Mozilla/5.0 (compatible; Xybernetex/1.0; +https://xybernetex.com)"
        ),
        "Accept": "text/html,application/xhtml+xml,*/*",
    })
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            raw = resp.read()
            content_type = resp.info().get("Content-Type", "")
            encoding_header = resp.info().get("Content-Encoding", "")
            if encoding_header == "gzip":
                import gzip
                raw = gzip.decompress(raw)
            # Detect charset
            charset = "utf-8"
            m = re.search(r"charset=([^\s;\"']+)", content_type, re.IGNORECASE)
            if m:
                charset = m.group(1)
            html = raw.decode(charset, errors="replace")
    except urllib.error.HTTPError as e:
        raise WebToolsError(f"HTTP {e.code} fetching {url}")
    except urllib.error.URLError as e:
        raise WebToolsError(f"Cannot fetch {url}: {e}")

    # Extract title
    title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    title = _strip_html(title_match.group(1)) if title_match else ""

    # Extract meta description
    desc_match = re.search(
        r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']*)["\']',
        html, re.IGNORECASE
    )
    description = desc_match.group(1) if desc_match else ""

    # Try to extract main content (prefer <main>, <article>, <body>)
    for tag in ("main", "article", "body"):
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", html, re.DOTALL | re.IGNORECASE)
        if m:
            content_html = m.group(1)
            break
    else:
        content_html = html

    content = _strip_html(content_html)
    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars]

    return {
        "url":         url,
        "title":       title,
        "description": description,
        "content":     content,
        "length":      len(content_html),
        "truncated":   truncated,
    }


# ══════════════════════════════════════════════════════════════════════════════
# N8N MODE — existing behaviour
# ══════════════════════════════════════════════════════════════════════════════

def _n8n_post(path: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """POST JSON to an n8n webhook, return parsed response."""
    url = f"{N8N_BASE_URL}/webhook/{path}"
    body = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if WEBHOOK_SECRET:
        headers["X-N8N-Webhook-Secret"] = WEBHOOK_SECRET

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")[:300]
        raise WebToolsError(f"n8n returned HTTP {e.code}: {body_text}")
    except urllib.error.URLError as e:
        raise WebToolsError(
            f"Cannot reach n8n at {url}\n"
            f"Is Tailscale connected? Is n8n running?\nError: {e}"
        )
    except json.JSONDecodeError as e:
        raise WebToolsError(f"n8n returned non-JSON: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — mode selected automatically
# ══════════════════════════════════════════════════════════════════════════════

def search_web(
    query: str,
    num_results: int = 8,
    tavily_api_key: str = "",
    brave_api_key: str = "",
) -> dict:
    """
    Search the web.  Priority: Tavily → Brave → n8n.

    Pass tavily_api_key or brave_api_key to use per-request credentials
    instead of the server's env vars.

    Returns:
        {
            "query": str,
            "total_found": int,
            "results": [{"title", "url", "snippet", "age", "source"}, ...],
            "answer": str  (Tavily only — empty string for other backends)
        }

    Raises WebToolsError on failure.
    """
    _tavily = tavily_api_key or TAVILY_API_KEY
    _brave  = brave_api_key  or BRAVE_API_KEY
    if _tavily:
        return _tavily_search(query, num_results, api_key=_tavily)
    if _brave:
        return _brave_search(query, num_results, api_key=_brave)
    if N8N_BASE_URL:
        return _n8n_post(SEARCH_PATH, {"query": query, "num_results": num_results})
    raise WebToolsError(
        "Web search not configured. Pass tavily_api_key or brave_api_key, "
        "or set TAVILY_API_KEY / BRAVE_API_KEY in .env."
    )


def fetch_url(
    url: str,
    max_chars: int = 8000,
    tavily_api_key: str = "",
    brave_api_key: str = "",
) -> dict:
    """
    Fetch a URL and return cleaned plain text.
    Priority: Tavily Extract → direct urllib fetch → n8n.

    Returns:
        {
            "url": str,
            "title": str,
            "description": str,
            "content": str,
            "length": int,
            "truncated": bool
        }

    Raises WebToolsError on failure.
    """
    _tavily = tavily_api_key or TAVILY_API_KEY
    _brave  = brave_api_key  or BRAVE_API_KEY
    if _tavily:
        return _tavily_extract(url, max_chars, api_key=_tavily)
    if _brave:
        return _direct_fetch(url, max_chars)
    if N8N_BASE_URL:
        return _n8n_post(FETCH_PATH, {"url": url, "max_chars": max_chars})
    raise WebToolsError(
        "Web fetch not configured. Pass tavily_api_key or brave_api_key, "
        "or set TAVILY_API_KEY / BRAVE_API_KEY in .env."
    )


def is_web_enabled() -> bool:
    """Returns True if any search mode is configured."""
    return bool(TAVILY_API_KEY or BRAVE_API_KEY or N8N_BASE_URL)


def get_search_backend() -> str:
    """Return the name of the active search backend (for logging)."""
    if TAVILY_API_KEY:
        return "tavily"
    if BRAVE_API_KEY:
        return "brave"
    if N8N_BASE_URL:
        return "n8n"
    return "none"


# ── Formatters (for agent context injection) ───────────────────────────────────

def format_search_results(data: dict) -> str:
    """Render search results as a compact readable block for the planner."""
    backend = get_search_backend()
    lines = [
        f"WEB SEARCH RESULTS — query: \"{data.get('query', '')}\"",
        f"({data.get('total_found', 0)} results found via {backend})",
        "",
    ]

    # Tavily AI answer — inject at top if available
    answer = data.get("answer", "")
    if answer:
        lines.append(f"AI SUMMARY: {answer[:600]}")
        lines.append("")

    for i, r in enumerate(data.get("results", []), 1):
        age = f"  [{r['age']}]" if r.get("age") else ""
        lines.append(f"{i}. {r['title']}{age}")
        lines.append(f"   {r['url']}")
        if r.get("snippet"):
            lines.append(f"   {r['snippet'][:300]}")
        lines.append("")
    return "\n".join(lines)


def format_fetch_result(data: dict) -> str:
    """Render a fetched page as a compact block for the planner."""
    truncation_note = "  [content truncated]" if data.get("truncated") else ""
    lines = [
        f"FETCHED URL: {data.get('url', '')}",
        f"TITLE: {data.get('title', '')}",
    ]
    if data.get("description"):
        lines.append(f"SUMMARY: {data['description']}")
    lines += [
        "",
        f"CONTENT:{truncation_note}",
        data.get("content", "(no content extracted)"),
    ]
    return "\n".join(lines)


def extract_urls_from_text(text: str) -> list[str]:
    """Pull URLs out of a planner focus/label field for FETCH_URL action."""
    pattern = r'https?://[^\s\'"<>]+'
    return re.findall(pattern, text)
