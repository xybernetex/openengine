# models/

This directory holds trained policy weights for the Xybernetex AI OpenEngine.

## Getting the base policy

The pre-trained `base_policy.pt` checkpoint is **not committed to this
repository** (model weights are excluded by `.gitignore` to keep the repo
size manageable).

Download the latest checkpoint from the **GitHub Releases** page:

> **https://github.com/xybernetex/openengine/releases**  *(placeholder — update once published)*

Place the downloaded file at:

```
models/base_policy.pt
```

The engine will load it automatically on startup.

## File layout

| File | Description |
|---|---|
| `base_policy.pt` | Base ActorCritic policy weights (download from Releases) |
| `fine_tuned_*.pt` | Optional fine-tuned checkpoints you train yourself |

## Training your own checkpoint

See [`docs/training.md`](../docs/training.md) for a full guide on running
behavioural cloning and PPO fine-tuning.

## IMPORTANT

**.pt files are never committed to git.**
The `.gitignore` at the repository root explicitly excludes `models/*.pt`.
Always distribute weights via GitHub Releases or a model registry such as
Hugging Face Hub — never via a git commit.
