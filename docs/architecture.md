# OpenEngine Architecture

## What is OpenEngine?

Xybernetex AI OpenEngine is a **reinforcement-learning policy combined with an
LLM execution layer**.  It is explicitly *not* a prompt loop, a chain-of-thought
scaffolding system, or a ReAct agent.

The distinction matters:

| Approach | Who decides WHAT to do next? | Who decides HOW to do it? |
|---|---|---|
| Prompt loop / ReAct | LLM (implicit in the text) | LLM |
| OpenEngine | **RL policy (ActorCritic network)** | **LLM (structured call)** |

The policy is a small neural network (~2 M parameters) trained via PPO.  It
observes a compact numeric state vector and outputs a categorical distribution
over 6 primitive actions.  The LLM is only called to fill in the *parameters*
for whichever action the policy chose — it never decides the action itself.

This separation means the policy can be trained to be efficient, safe, and
goal-directed without paying the latency and cost of a full LLM forward pass
on every decision.

---

## MDP Formulation

OpenEngine models goal execution as a **finite-horizon Markov Decision Process**:

### State space S

The observation vector is a fixed-length float32 tensor (currently `STATE_DIM`
dimensions) encoding:

- Intent class (one-hot over the goal taxonomy)
- Number of goal aspects identified / completed / remaining
- Step count and remaining step budget
- Was the last action successful? (binary)
- Tool availability flags (one per registered tool)
- Working-memory signals: has plan, has artifact, last action index

### Action space A

Six discrete primitive actions (see next section).

### Reward R

Default reward signal (see `training/reward.py` and `docs/training.md`):

| Event | Signal |
|---|---|
| Every step | −0.1 (efficiency incentive) |
| New aspect completed | +0.9 per aspect |
| Successful execution | +0.3 |
| DONE with task complete | +15.0 |
| DONE with task incomplete | −5.0 |
| Stagnation (no progress, no success) | −0.5 |

The reward function is **pluggable** — subclass `RewardFunction` and pass
your instance to the trainer to override the default.

### Horizon

Each episode has a configurable step budget (default: 20).  The episode
terminates when the policy takes the `DONE` action or the budget is exhausted.

---

## The Six Primitive Actions

```
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Action          │ Semantics                                                │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ PLAN            │ Ask the LLM to decompose the goal into structured        │
│                 │ aspects and store them in working memory.                │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ ANALYZE         │ Ask the LLM to evaluate current progress against the     │
│                 │ aspects and update completion status.                    │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ INVOKE_TOOL     │ Call a registered external tool via the ToolServer       │
│                 │ webhook. LLM generates the tool name and parameters.     │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ EXECUTE_CODE    │ Generate and run a Python code snippet in the sandboxed  │
│                 │ executor. Output is captured into working memory.        │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ WRITE_ARTIFACT  │ Ask the LLM to compose a structured output artifact      │
│                 │ (report, email draft, JSON, etc.) and store it.          │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ DONE            │ Signal that the goal is complete. Triggers terminal      │
│                 │ reward and ends the episode.                             │
└─────────────────┴──────────────────────────────────────────────────────────┘
```

The policy selects actions by sampling from the softmax output of the
`ActorCritic` network.  During inference a temperature parameter can be used
to make the policy more or less exploratory.

---

## Two-Level Hierarchy

```
                    ┌───────────────────────────────┐
                    │        Goal (plain text)       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      ActorCritic Policy        │
                    │   (decides WHICH action)       │
                    │   ──────────────────────────   │
                    │   input : state vector (S)     │
                    │   output: action logits (6)    │
                    └───────────────┬───────────────┘
                                    │  action index
                    ┌───────────────▼───────────────┐
                    │        LiveExecutor            │
                    │   (dispatches the action)      │
                    └──┬───────────┬───────────┬────┘
                       │           │           │
              ┌────────▼──┐  ┌────▼────┐  ┌───▼──────────┐
              │ LLMClient  │  │ToolSvr  │  │CodeExecutor  │
              │ (HOW logic)│  │Webhook  │  │ (sandboxed)  │
              └────────┬──┘  └────┬────┘  └───┬──────────┘
                       │           │           │
                    ┌──▼───────────▼───────────▼──┐
                    │       WorkingMemory          │
                    │       WorldEngine            │
                    └─────────────────────────────┘
```

The LLM is only invoked inside the executor to fill action parameters.  The
policy network never sees raw text — it operates on the numeric state vector.

---

## Capability Manifest System

A capability manifest is a JSON file that declares which tools are available
for a given run.  The engine reads the manifest at startup and registers each
tool with the `ToolRegistry`.

Example manifest structure:

```json
{
  "manifest_version": "1.0",
  "name": "my_workflow",
  "tools": [
    {
      "name": "web_search",
      "description": "Search the web via Tavily.",
      "webhook_action": "web_search",
      "parameters": { ... }
    }
  ]
}
```

The `webhook_action` field maps each tool to a ToolServer action endpoint.
See `examples/manifests/` for complete examples, and `docs/tools.md` for the
full manifest specification.

---

## ToolServer Plugin Architecture

The ToolServer is a lightweight HTTP service (FastAPI) that the engine calls
when it takes the `INVOKE_TOOL` action.  Each tool is a plugin module that
implements a single function:

```python
async def invoke(params: dict) -> ToolResult: ...
```

Tools are loaded from the `toolserver/plugins/` directory at startup and
registered by name.  The engine sends a webhook POST to
`XYBER_TOOL_WEBHOOK_URL` with a JSON body containing `action` and `params`.

See `docs/tools.md` for a complete plugin authoring guide.

---

## Episode Flow (ASCII Sequence)

```
User / API                Engine                 LLM          Tool / Code
    │                       │                     │                │
    │── goal + manifest ────▶│                     │                │
    │                       │─── vectorise state ─▶│                │
    │                       │                     │                │
    │                       │◀─── action: PLAN ───│                │
    │                       │─── PLAN prompt ─────▶│                │
    │                       │◀─── aspects JSON ────│                │
    │                       │─── update WM ────────│                │
    │                       │                     │                │
    │                       │◀─ action: INVOKE ────│                │
    │                       │─── tool params ─────▶│                │
    │                       │◀─── tool + params ───│                │
    │                       │──── webhook POST ────────────────────▶│
    │                       │◀─── tool result ─────────────────────│
    │                       │─── update WM ────────│                │
    │                       │                     │                │
    │                       │◀─ action: WRITE ─────│                │
    │                       │─── write prompt ────▶│                │
    │                       │◀─── artifact text ───│                │
    │                       │─── store artifact ───│                │
    │                       │                     │                │
    │                       │◀─ action: DONE ──────│                │
    │◀─── result dict ──────│                     │                │
```
