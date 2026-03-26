# Xybernetex AI OpenEngine

**A reinforcement learning policy engine for autonomous AI agents.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

Most "AI agents" are LLMs wrapped in a for loop. OpenEngine is different: a trained reinforcement learning policy controls *when* and *what* action to take, while the LLM handles language tasks. The policy learns from experience. A loop doesn't.

---

## How It Works

```
  Goal text
      │
      ▼
┌─────────────────────────────────────┐
│        Policy Network               │
│  ActorCritic (256-dim state → 6)    │
│  Trunk: 256→256→128 (GELU+LayerNorm)│
│  Actor head  → action logits        │
│  Critic head → value estimate       │
└────────────────┬────────────────────┘
                 │  discrete action (argmax + mask)
                 ▼
        ┌────────────────┐
        │  Live Executor │  ◄── per-action handlers
        └───────┬────────┘
                │
     ┌──────────┴───────────┐
     │  Which action?       │
     ├──────────────────────┤
     │ PLAN           → LLM decomposes goal into aspects
     │ ANALYZE        → LLM scans working memory, finds gaps
     │ INVOKE_TOOL    → LLM picks tool from manifest → webhook
     │ EXECUTE_CODE   → sandbox runs latest CODE artifact
     │ WRITE_ARTIFACT → LLM writes CODE or DOCUMENT to disk
     │ DONE           → termination gate checks all aspects
     └──────────────────────┘
                │
                ▼
       Result + reward signal
                │
                └──► state vector updated → back to policy
```

**The key split:** the policy never generates text. The LLM never decides what to do next. Each is doing only what it is good at.

`INVOKE_TOOL` is a meta-action: the policy selects it, then the LLM router reads the capability manifest and picks which registered tool to call. Adding a new tool never requires retraining the policy.

---

## Quick Start

**Prerequisites:** Python 3.11+, pip

```bash
# 1. Clone and set up environment
git clone https://github.com/xybernetex/openengine.git
cd openengine
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure credentials
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the relevant API key (see LLM Providers below)

# 3. Download the base policy checkpoint
# Grab base_policy.pt from the latest GitHub Release and place it in models/

# 4. Run a goal
cd engine
python inference_worker.py "Research the latest advances in fusion energy and write a summary report"
```

**Expected terminal output:**

```
╔══════════════════════════════════════════════════════════════════╗
║  XYBERNETEX AI ENGINE  ·  run_id: a3f9c1b2                      ║
║  Goal: Research the latest advances in fusion energy...          ║
╚══════════════════════════════════════════════════════════════════╝

 STEP 1  ·  PLAN
 ─────────────────────────────────────────────────────────────────
 Decomposing goal into tracked aspects...
 ✔  aspects registered: [recent_breakthroughs, key_players,
    timeline, summary_report]
 reward: +0.00

 STEP 2  ·  INVOKE_TOOL  →  web_search
 ─────────────────────────────────────────────────────────────────
 Query: "fusion energy advances 2024 2025"
 ✔  4 results returned, answer synthesised
 reward: +0.20

 STEP 3  ·  ANALYZE
 ─────────────────────────────────────────────────────────────────
 Scanning working memory... 2 aspects in_progress, 2 pending
 reward: +0.00

 STEP 4  ·  WRITE_ARTIFACT  →  DOCUMENT
 ─────────────────────────────────────────────────────────────────
 Writing: fusion_energy_summary.md
 ✔  artifact persisted  (1 842 chars)
 reward: +1.00  [aspect complete: summary_report]

 STEP 5  ·  DONE
 ─────────────────────────────────────────────────────────────────
 Termination gate: all 4 aspects complete ✔
 reward: +15.00

═══════════════════════════════════════════════════════════════════
 COMPLETE  ·  5 steps  ·  total reward: +16.20
═══════════════════════════════════════════════════════════════════
```

---

## Architecture

### The MDP

The agent operates as a Markov Decision Process:

| Component | Details |
|---|---|
| **State** | 256-dim float32 vector — aspect completion, action history (32 steps × 6 actions), artifact counts, error telemetry, step budget, intent signals |
| **Actions** | 6 discrete primitives (see below) |
| **Reward** | Shaped: +15.0 valid DONE, +1.0 aspect completion, +0.5 exec success, −0.1 step penalty, −0.75 stagnation, −5.0 premature DONE |

**Action space:**

| Index | Primitive | What it does |
|---|---|---|
| 0 | `PLAN` | Decompose the goal into aspects with verifiable completion criteria |
| 1 | `ANALYZE` | Scan working memory, identify gaps, determine highest-value next action |
| 2 | `INVOKE_TOOL` | Meta-action: LLM selects a tool from the manifest and calls it |
| 3 | `EXECUTE_CODE` | Run the most recent `CODE` artifact in a sandboxed subprocess |
| 4 | `WRITE_ARTIFACT` | Generate a `CODE` or `DOCUMENT` artifact via LLM and persist to disk |
| 5 | `DONE` | Assert task completion — validated by checking all aspects are complete |

### Two-Level Hierarchy

```
Control Plane (RL policy)          Execution Plane (LLM)
──────────────────────────         ─────────────────────────────
ActorCritic network                LLMClient (multi-provider)
Selects action index               Generates text for that action
Sees 256-dim state vector          Sees goal + working memory context
Trained offline, frozen at run     Stateless per call
No text generation                 No action selection
```

The policy runs in microseconds. LLM calls are only made when the policy selects an action that requires language — `PLAN`, `ANALYZE`, `INVOKE_TOOL`, and `WRITE_ARTIFACT`.

### Capability Manifest

Tools are declared in a `CapabilityManifest` — a JSON structure describing what the agent is authorized to call in a given run:

```json
{
  "run_id": "a3f9c1b2",
  "user_id": "local",
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web via Tavily",
      "actions": ["search"],
      "input_schema": { "type": "object" },
      "risk": "low",
      "read_only": true,
      "backend": "webhook"
    },
    {
      "name": "send_email",
      "description": "Send email via Resend",
      "actions": ["send"],
      "risk": "medium",
      "read_only": false,
      "requires_approval": true,
      "backend": "webhook"
    }
  ]
}
```

When the policy selects `INVOKE_TOOL`, the LLM router reads this manifest and decides which tool to call based on the current task context. The policy never sees tool names — it just knows a tool call is appropriate. This is what makes tool addition retraining-free.

### ToolServer

ToolServer is a FastAPI webhook listener that serves as the execution backend for `INVOKE_TOOL`. It auto-discovers tool modules from the `toolserver/tools/` directory at startup — drop in a Python file with a `run()` function and it is immediately available.

```
ToolServer (FastAPI, port 8001)
├── POST /invoke        ← receives ToolInvocation envelopes
├── GET  /health        ← liveness check
└── tools/
    ├── web_search.py   ← Tavily search
    └── send_email.py   ← Resend email
```

Any webhook-callable endpoint can be a tool: N8N workflows, custom REST APIs, internal services. The agent does not care what is behind the webhook.

---

## Adding Tools

A minimal tool is a single async function in `toolserver/tools/`:

```python
# toolserver/tools/get_weather.py

async def run(invocation: dict) -> dict:
    city = invocation.get("input", {}).get("city", "London")
    # ... call your weather API here ...
    return {
        "status": "succeeded",
        "output": {"temperature_c": 18, "conditions": "cloudy"},
        "error": "",
    }
```

Then add a descriptor to your manifest:

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "actions": ["get"],
  "risk": "low",
  "read_only": true,
  "backend": "webhook"
}
```

Restart ToolServer. The tool is live. No policy retraining required — `INVOKE_TOOL` already handles it.

---

## Training Your Own Policy

The base checkpoint shipped in releases was trained on ~50k synthetic behavioral cloning demonstrations, then fine-tuned for ~1M PPO steps. You can extend it or train from scratch.

**Two-phase training:**

1. **Behavioral Cloning** — warm-start from oracle demonstrations generated by `training/rollout_generator.py`
2. **PPO Fine-tuning** — policy gradient training via `training/ppo_trainer.py` against the `XybernetexEnv` gym environment

```bash
# Generate demonstrations
python training/rollout_generator.py --goals 10000 --output data/demos.jsonl

# Behavioral cloning warm-start
python training/ppo_trainer.py --mode bc --demos data/demos.jsonl --epochs 5

# PPO fine-tuning
python training/ppo_trainer.py \
    --init-from models/bc_policy.pt \
    --rollout-steps 2048 \
    --n-envs 8 \
    --total-steps 1000000 \
    --output models/ppo_policy.pt
```

Full guide: [`docs/training.md`](docs/training.md)

---

## LLM Providers

Set `LLM_PROVIDER` in your `.env` to switch providers. `LLM_API_KEY` applies to all providers except Cloudflare (which uses `CF_ACCOUNT_ID` + `CF_API_TOKEN`).

| Provider | `LLM_PROVIDER` value | Default model | Notes |
|---|---|---|---|
| Cloudflare Workers AI | `cloudflare` | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | No `LLM_API_KEY` needed |
| OpenAI | `openai` | `gpt-4o` | Requires `LLM_API_KEY` |
| Mistral | `mistral` | `mistral-large-latest` | Requires `LLM_API_KEY` |
| Anthropic | `anthropic` | `claude-opus-4-6` | Requires `LLM_API_KEY` |
| Google Gemini | `gemini` | `gemini-2.0-flash` | Requires `LLM_API_KEY` |

Override the model with `LLM_MODEL=<model-id>` in your `.env`.

---

## Project Structure

```
openengine/
├── engine/                  # Core inference loop
│   ├── inference_worker.py  # Entry point — runs a goal to completion
│   ├── live_executor.py     # Action dispatcher (PLAN/ANALYZE/etc → handlers)
│   ├── rl_chassis.py        # ActorCritic, ActionSpace, StateVectorizer, RewardFunction
│   ├── xyber_env.py         # OpenAI Gym-compatible environment
│   ├── llm_client.py        # Multi-provider LLM abstraction
│   ├── terminal.py          # Bloomberg-style terminal display
│   └── tick_logger.py       # Structured per-step logging
├── core/                    # Shared runtime libraries
│   ├── working_memory.py    # SQLite-backed per-run state store
│   ├── world_engine.py      # Goal/aspect lifecycle management
│   ├── code_executor.py     # Sandboxed subprocess runner
│   └── web_tools.py         # Web search helpers
├── connectors/              # Tool contract types
│   ├── contracts.py         # ToolInvocation, ToolResult, ToolGateway
│   └── registry.py          # CapabilityManifest, ToolDescriptor, ToolRegistry
├── toolserver/              # FastAPI webhook execution layer
│   ├── main.py              # Server entrypoint
│   ├── registry.py          # Plugin autodiscovery
│   └── tools/
│       ├── web_search.py    # Tavily web search
│       └── send_email.py    # Resend email
├── training/                # Policy training pipeline
│   ├── rollout_generator.py # Synthetic BC demonstration generator
│   ├── ppo_trainer.py       # PPO + BC training loop
│   ├── reward.py            # Pluggable reward interface
│   └── xyber_env.py         # Training environment
├── models/                  # Policy checkpoints (base_policy.pt from releases)
├── schemas/                 # Database schema
├── docs/                    # Extended documentation
│   ├── architecture.md
│   ├── training.md
│   ├── llm_providers.md
│   └── tools.md
└── requirements.txt
```

---

## Contributing

OpenEngine is licensed under the **GNU Affero General Public License v3.0**. This means contributions flow back to the community — if you run a modified version as a service, your modifications must be open source under the same terms.

**Ways to contribute:**

- **Bug reports** — open a GitHub issue with reproduction steps
- **New tool plugins** — add a file to `toolserver/tools/` (see Adding Tools above)
- **New LLM providers** — extend `engine/llm_client.py`
- **Training improvements** — better goal taxonomies, reward shaping, new goal classes
- **PRs** — all changes welcome; run `pytest` and `ruff check .` before submitting

When adding a tool, include a descriptor JSON and a one-sentence description of what it does in your PR description.

---

## Managed Cloud

[**xybernetex.com**](https://xybernetex.com) offers a hosted version of OpenEngine with:

- Authentication and multi-user workspaces
- Billing and usage metering
- Scheduled and triggered agent runs
- The Workspace UI for browsing artifacts, run history, and agent traces
- Managed ToolServer with pre-built integrations

The core engine in this repo is the same engine that powers the hosted platform.

---

## License

**GNU Affero General Public License v3.0 (AGPL-3.0)**

You are free to use, modify, and distribute this software. If you run a modified version over a network, you must make the source available under the same license. See [`LICENSE`](LICENSE) for the full terms.
