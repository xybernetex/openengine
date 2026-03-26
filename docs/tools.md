# Tool Development Guide

This document explains the ToolServer architecture, how to write a new tool
plugin, the ToolInvocation / ToolResult contract, the capability manifest
format, and how to integrate n8n workflows as tool backends.

---

## ToolServer Architecture

The ToolServer is a FastAPI service that acts as a bridge between the engine
and external services.  When the engine takes the `INVOKE_TOOL` action, it
sends an HTTP POST to the webhook URL configured in `.env`:

```
XYBER_TOOL_WEBHOOK_URL=http://localhost:9000/invoke
```

The request body is a JSON object with two fields:

```json
{
  "action": "web_search",
  "params": {
    "query": "open source AI agents 2025",
    "max_results": 5
  }
}
```

The ToolServer routes the request to the matching plugin, executes it, and
returns a standardised `ToolResult` JSON object.

### Request authentication

All requests include an HMAC-SHA256 signature in the `X-Xyber-Signature`
header, computed over the raw request body using `XYBER_TOOL_WEBHOOK_SECRET`.
Plugins should verify this signature before processing.

---

## Writing a New Tool Plugin

A tool plugin is a Python module placed in `toolserver/plugins/`.  It must
expose a single async function:

```python
async def invoke(params: dict) -> dict: ...
```

The function receives the `params` dict from the webhook request and must
return a dict that will be serialised as the `ToolResult.data` field.

### Full example: `my_weather_tool.py`

```python
"""
toolserver/plugins/my_weather_tool.py

Fetches current weather data from Open-Meteo (no API key required).
Register this tool in your capability manifest with:
  "webhook_action": "get_weather"
"""
from __future__ import annotations

import httpx

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


async def invoke(params: dict) -> dict:
    """
    Fetch current weather for a given latitude / longitude.

    params
    ------
    latitude  : float  — required
    longitude : float  — required
    """
    lat = float(params["latitude"])
    lon = float(params["longitude"])

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "current_weather": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current_weather", {})
    return {
        "temperature_c": current.get("temperature"),
        "windspeed_kmh": current.get("windspeed"),
        "weathercode":   current.get("weathercode"),
        "latitude":      lat,
        "longitude":     lon,
    }
```

### Registering the plugin

Plugins are auto-discovered by the ToolServer at startup.  The webhook
`action` name is derived from the module filename (without `.py`) unless you
override it via a top-level `ACTION_NAME` constant:

```python
ACTION_NAME = "get_weather"   # optional — defaults to module name
```

No further registration code is required.

---

## ToolInvocation / ToolResult Contract

### ToolInvocation (inbound)

| Field    | Type   | Description |
|---|---|---|
| `action` | string | The tool name / webhook action key. |
| `params` | object | Arbitrary key-value parameters for the tool. |

### ToolResult (outbound)

| Field      | Type    | Description |
|---|---|---|
| `success`  | boolean | Whether the tool executed without error. |
| `data`     | object  | Arbitrary return value from the plugin. |
| `error`    | string  | Error message (present only when `success` is false). |
| `duration` | number  | Wall-clock execution time in seconds. |

Example successful response:

```json
{
  "success": true,
  "data": {
    "temperature_c": 18.2,
    "windspeed_kmh": 12.0,
    "weathercode": 2
  },
  "duration": 0.34
}
```

Example error response:

```json
{
  "success": false,
  "data": {},
  "error": "Connection timeout after 10 s",
  "duration": 10.01
}
```

---

## Capability Manifest Format

The capability manifest tells the engine which tools are available for a
given run.  It is a JSON file with the following top-level structure:

```jsonc
{
  "manifest_version": "1.0",
  "name": "my_workflow",          // optional human-readable name
  "description": "...",           // optional description
  "tools": [ ... ]                // array of tool definitions
}
```

### Tool definition schema

```jsonc
{
  "name": "tool_name",             // must match webhook_action
  "description": "...",            // shown to the LLM when selecting params
  "webhook_action": "tool_name",   // POST action key sent to ToolServer
  "parameters": {
    "param_name": {
      "type": "string|integer|number|boolean|array|object",
      "description": "...",
      "required": true,
      "default": null              // optional
    }
  },
  "returns": {                     // optional — documents the return shape
    "type": "object",
    "properties": { ... }
  }
}
```

### Loading a manifest at runtime

```python
from engine.inference_worker import run_local_goal

result = run_local_goal(
    goal="Search the web for AI news and email me a summary.",
    capability_manifest_path="examples/manifests/research_and_email.json",
)
```

See `examples/manifests/` for ready-to-use examples.

---

## Registering Tools with the Engine

The `ToolRegistry` (in `connectors/`) is populated from the capability
manifest at the start of each run.  You do not need to manually register tools
in code — the manifest is the single source of truth.

If you want to use a tool in every run regardless of manifest, add it to
`DEFAULT_TOOLS` in `connectors/__init__.py`.

---

## n8n Connector Usage

n8n is supported as a ToolServer backend for teams that prefer a visual
workflow builder over writing Python plugins.

### Setup

1. Deploy n8n and create a workflow with a **Webhook** trigger node.
2. Set the webhook URL in your `.env`:
   ```
   XYBER_TOOL_WEBHOOK_URL=https://your-n8n-host/webhook/xyber-tools
   ```
3. In n8n, add a **Switch** node that routes on `{{ $json.action }}` to
   different sub-workflows, one per tool.
4. Each sub-workflow should read from `{{ $json.params }}` and return a
   `ToolResult`-compatible JSON body.

### Shared secret

n8n can verify the `X-Xyber-Signature` header using its **Header Auth**
credential or a **Code** node that re-computes the HMAC:

```javascript
const crypto = require("crypto");
const secret = $env.XYBER_TOOL_WEBHOOK_SECRET;
const body = JSON.stringify($input.all()[0].json);
const sig = crypto.createHmac("sha256", secret).update(body).digest("hex");
if (sig !== $input.all()[0].headers["x-xyber-signature"]) {
  throw new Error("Invalid signature");
}
return $input.all();
```

### Limitations

- n8n webhooks are HTTP-only within the engine's request cycle (no
  server-sent events or streaming).
- Long-running n8n workflows (>30 s) should use async patterns and a
  callback URL rather than a synchronous response.
