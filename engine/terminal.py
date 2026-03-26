"""
terminal.py — Xybernetex Debug Output Layer

Stripped-down replacement for the Bloomberg-style animated terminal.
Zero animations, zero sleeps, zero LLM calls.  Every function keeps
the same signature as the original so inference_worker.py is unchanged.

Output format:  plain structured lines → fast, grep-friendly, accurate.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── WINDOWS UTF-8 ─────────────────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ANSI COLOUR CONSTANTS (kept so inference_worker.py can use T.RED etc.) ────
RESET   = "\033[0m"
BOLD    = "\033[1m"
DIM     = "\033[2m"
RED     = "\033[91m"
GREEN   = "\033[92m"
AMBER   = "\033[33m"
YELLOW  = "\033[93m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
WHITE   = "\033[97m"
GRAY    = "\033[90m"

WIDTH = 72

# ── ACTION METADATA ───────────────────────────────────────────────────────────
ACTION_COLORS = {
    "PLAN"          : CYAN,
    "ANALYZE"       : YELLOW,
    "INVOKE_TOOL"   : MAGENTA,
    "EXECUTE_CODE"  : GREEN,
    "WRITE_ARTIFACT": WHITE,
    "DONE"          : GREEN + BOLD,
}

ACTION_DESCRIPTIONS = {
    "PLAN"          : "Decompose goal into tracked aspects.",
    "ANALYZE"       : "Scan working memory, identify gaps.",
    "INVOKE_TOOL"   : "Invoke a registered tool via webhook.",
    "EXECUTE_CODE"  : "Run the generated code artifact.",
    "WRITE_ARTIFACT": "Generate a CODE or DOCUMENT artifact.",
    "DONE"          : "Validate completion and terminate.",
}

# ── REWARD HELPERS ────────────────────────────────────────────────────────────

def reward_color(val: float) -> str:
    if val >= 10.0: return GREEN + BOLD
    if val >= 0.5:  return GREEN
    if val >= 0.0:  return YELLOW
    if val >= -1.0: return AMBER
    return RED

def reward_label(val: float) -> str:
    if val >= 10.0: return "TERMINAL_SUCCESS"
    if val >= 0.8:  return "POSITIVE"
    if val >= 0.3:  return "EXEC_SUCCESS"
    if val > -0.2:  return "STAGNATION"
    if val > -1.0:  return "STAGNATION"
    if val <= -4.0: return "DONE_INVALID"
    return "PENALTY"

# ── PRIMITIVES ────────────────────────────────────────────────────────────────

def write(text: str, end: str = "\n", flush: bool = True) -> None:
    print(text, end=end, flush=flush)

def blank() -> None:
    print("", flush=True)

def divider(char: str = "─", width: int = WIDTH, color: str = GRAY) -> None:
    print(f"  {color}{char * width}{RESET}", flush=True)

# No-op stubs for functions that existed but are animation-only
def typewrite(text: str, delay: float = 0.0, color: str = "", end: str = "\n") -> None:
    write(f"{color}{text}{RESET}", end=end)

def typewrite_line(prefix: str, text: str, prefix_color: str = GRAY, text_color: str = WHITE) -> None:
    print(f"{prefix_color}{prefix}{RESET}{text_color}{text}{RESET}", flush=True)

def flash(text: str, color: str = "", times: int = 1, interval: float = 0.0) -> None:
    write(f"{color}{text}{RESET}")

def header_box(lines: list, color: str = GREEN) -> None:
    divider("─", color=color)
    for line in lines:
        write(f"  {color}{line}{RESET}")
    divider("─", color=color)

# ── WORKING MEMORY BRIDGE ─────────────────────────────────────────────────────

class WorkingMemoryBridge:
    _wm     = None
    _run_id = None

    @classmethod
    def register(cls, wm, run_id: str) -> None:
        cls._wm     = wm
        cls._run_id = run_id

    @classmethod
    def get_aspects(cls) -> list:
        if cls._wm is None or cls._run_id is None:
            return []
        try:
            return cls._wm.get_goal_aspects(cls._run_id)
        except Exception:
            return []

    @classmethod
    def get_artifacts(cls) -> list:
        if cls._wm is None or cls._run_id is None:
            return []
        try:
            return cls._wm.get_artifacts(cls._run_id)
        except Exception:
            return []

bridge = WorkingMemoryBridge()

# ── EVENT BRIDGE ──────────────────────────────────────────────────────────────

class EventBridge:
    _publisher = None
    _run_id    = None

    @classmethod
    def register(cls, publisher, run_id) -> None:
        cls._publisher = publisher
        cls._run_id    = run_id

    @classmethod
    def clear(cls) -> None:
        cls._publisher = None
        cls._run_id    = None

    @classmethod
    def publish(cls, event_data: dict) -> None:
        if cls._publisher is None or not cls._run_id:
            return
        try:
            cls._publisher(cls._run_id, event_data)
        except Exception:
            pass

event_bridge = EventBridge()

def set_event_sink(publisher, run_id=None) -> None:
    event_bridge.register(publisher, run_id)

def clear_event_sink() -> None:
    event_bridge.clear()

# ── COGNITIVE STREAM — no-op stub ─────────────────────────────────────────────
# The original fired a background LLM call purely for atmospheric text.
# Replaced with a do-nothing object that satisfies all call sites.

class CognitiveStream:
    def __init__(self, *args, **kwargs):
        pass
    def start(self) -> None:
        pass
    def stop(self) -> None:
        pass
    def get_lines(self) -> list:
        return []

# ── LIVE REASONING — debug print instead of spinner ──────────────────────────

class LiveReasoning:
    def __init__(self, action_name: str, run_id: str = "", cognitive_stream=None):
        self.action_name = action_name
        self.run_id      = run_id

    def start(self) -> None:
        pass   # step_announce already printed the header

    def stop(self) -> None:
        pass

# ── BOOT SEQUENCE ─────────────────────────────────────────────────────────────

def boot_sequence(goal: str, model_path: str, run_id: str | None = None) -> None:
    blank()
    divider("═")
    write(f"  {CYAN}{BOLD}XYBERNETEX  //  PPO INFERENCE ENGINE{RESET}")
    divider("═")
    write(f"  {GRAY}RUN ID   {RESET}{CYAN}{run_id or 'N/A'}{RESET}")
    write(f"  {GRAY}MODEL    {RESET}{WHITE}{model_path}{RESET}")
    write(f"  {GRAY}GOAL     {RESET}{WHITE}{goal}{RESET}")
    divider("─")
    blank()

# ── STEP ANNOUNCE ─────────────────────────────────────────────────────────────

def step_announce(
    step_num: int,
    max_steps: int,
    action_name: str,
    prev_reward: float | None = None,
) -> None:
    ac   = ACTION_COLORS.get(action_name, WHITE)
    desc = ACTION_DESCRIPTIONS.get(action_name, "")

    if prev_reward is not None:
        rc = reward_color(prev_reward)
        rl = reward_label(prev_reward)
        rew_str = f"prev_reward={rc}{prev_reward:+.4f}{RESET}  {GRAY}{rl}{RESET}"
    else:
        rew_str = f"{GRAY}prev_reward=—{RESET}"

    divider("─")
    write(
        f"  {GRAY}STEP {step_num:02d}/{max_steps}{RESET}  "
        f"{ac}{BOLD}{action_name}{RESET}  │  {rew_str}"
    )
    write(f"  {GRAY}{desc}{RESET}")

    event_bridge.publish({
        "type": "narrative",
        "data": {"text": f"Step {step_num}/{max_steps}: {action_name} - {desc}"},
    })

# ── REWARD REVEAL ─────────────────────────────────────────────────────────────

def reveal_reward(reward_val: float, step_num: int, cumulative_reward: float) -> None:
    rc  = reward_color(reward_val)
    rl  = reward_label(reward_val)
    cc  = reward_color(cumulative_reward)

    write(
        f"  {GRAY}>{RESET} REWARD  "
        f"{rc}{BOLD}{reward_val:+.4f}{RESET}  "
        f"{GRAY}{rl}  │  cumulative={RESET}{cc}{cumulative_reward:+.4f}{RESET}"
    )

    # Aspect panel inline
    aspects = bridge.get_aspects()
    if aspects:
        blank()
        complete = sum(1 for a in aspects if a.get("status") == "complete")
        total    = len(aspects)
        bar_len  = 24
        filled   = int(bar_len * complete / total) if total else 0
        bar      = "█" * filled + "░" * (bar_len - filled)
        write(f"  {GRAY}ASPECTS  [{bar}]  {complete}/{total}  {int(complete/total*100)}%{RESET}")
        for a in aspects:
            status = a.get("status", "pending")
            sc = GREEN if status == "complete" else (AMBER if status == "in_progress" else GRAY)
            sym = "✓" if status == "complete" else ("~" if status == "in_progress" else "○")
            write(f"    {sc}{sym}  {a.get('aspect','?'):<40}  {status}{RESET}")

    event_bridge.publish({
        "type": "reward",
        "data": {"reward": reward_val, "cumulative": cumulative_reward},
    })

# ── ASPECT PANEL (standalone) ─────────────────────────────────────────────────

def show_aspect_panel() -> None:
    aspects = bridge.get_aspects()
    if not aspects:
        return
    divider("─")
    for a in aspects:
        status = a.get("status", "pending")
        sc = GREEN if status == "complete" else (AMBER if status == "in_progress" else GRAY)
        sym = "✓" if status == "complete" else ("~" if status == "in_progress" else "○")
        write(f"    {sc}{sym}  {a.get('aspect','?'):<40}  {status}{RESET}")

# ── CODE DISPLAY ──────────────────────────────────────────────────────────────

def stream_code_extraction(code_content: str) -> None:
    divider("─")
    write(f"  {CYAN}ARTIFACT CODE{RESET}")
    divider("─")
    lines = code_content.splitlines()
    cap   = 60
    for i, line in enumerate(lines[:cap]):
        write(f"  {GRAY}{i+1:>3}{RESET}  {line.rstrip()}")
    if len(lines) > cap:
        write(f"  {GRAY}[{len(lines)-cap} more lines]{RESET}")
    divider("─")

# ── SUBPROCESS EXECUTION (functional — kept as-is, just no animations) ────────

def stream_execution_live(script_path: str, timeout: int = 30) -> dict:
    name = Path(script_path).name
    blank()
    divider("─")
    write(f"  {GREEN}EXECUTE{RESET}  {WHITE}{name}{RESET}  {GRAY}{script_path}{RESET}")
    divider("─")

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    returncode   = -1
    start_time   = time.time()

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:
            clean = line.rstrip()
            write(f"  {CYAN}│{RESET}  {clean}")
            stdout_lines.append(clean)

        remaining = max(1, timeout - int(time.time() - start_time))
        proc.wait(timeout=remaining)
        returncode = proc.returncode

        raw_err = proc.stderr.read()
        if raw_err:
            stderr_lines = raw_err.splitlines()

    except subprocess.TimeoutExpired:
        try:    proc.kill()
        except Exception: pass
        returncode   = -1
        stderr_lines = [f"[TIMEOUT after {timeout}s]"]
    except Exception as exc:
        returncode   = -1
        stderr_lines = [str(exc)]

    elapsed = time.time() - start_time
    success = returncode == 0
    sc      = GREEN if success else RED
    write(f"  {sc}{'SUCCESS' if success else 'FAILED'}{RESET}  "
          f"{GRAY}exit={returncode}  elapsed={elapsed:.2f}s{RESET}")

    if stderr_lines:
        write(f"  {RED}STDERR:{RESET}")
        for ln in stderr_lines[:20]:
            write(f"    {RED}{ln}{RESET}")

    blank()
    return {
        "success"   : success,
        "returncode": returncode,
        "stdout"    : "\n".join(stdout_lines),
        "stderr"    : "\n".join(stderr_lines),
        "elapsed"   : elapsed,
    }

# ── TICK + LOG WRITERS ────────────────────────────────────────────────────────

def write_tick(
    ticks_root: Path,
    run_id: str,
    step_num: int,
    action_name: str,
    data: dict,
) -> Path:
    path = ticks_root / run_id / f"step_{step_num:02d}_{action_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data["_meta"] = {
        "run_id"   : run_id,
        "step"     : step_num,
        "action"   : action_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path

def write_run_log(logs_root: Path, run_id: str, summary: dict) -> Path:
    path = logs_root / run_id / "run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path

# ── RUN COMPLETE ──────────────────────────────────────────────────────────────

def run_complete(
    total_steps: int,
    total_reward: float,
    success: bool,
    run_id: str,
    log_path: Path,
    goal: str,
) -> None:
    ec = GREEN if success else AMBER
    rc = reward_color(total_reward)
    blank()
    divider("═", color=ec)
    write(f"  {ec}{'GOAL ACHIEVED' if success else 'MAX STEPS — INCOMPLETE'}{RESET}")
    divider("─")
    write(f"  {GRAY}STATUS    {RESET}{ec}{'COMPLETE' if success else 'TIMEOUT'}{RESET}")
    write(f"  {GRAY}STEPS     {RESET}{WHITE}{total_steps}/25{RESET}")
    write(f"  {GRAY}REWARD    {RESET}{rc}{total_reward:+.4f}{RESET}")
    write(f"  {GRAY}RUN ID    {RESET}{CYAN}{run_id}{RESET}")
    write(f"  {GRAY}LOG       {RESET}{GRAY}{log_path}{RESET}")
    divider("═", color=ec)
    blank()

# ── MATRIX RAIN — removed; stub kept for any stale import references ──────────

def matrix_rain(*args, **kwargs) -> None:
    pass

def get_fragments(action_type: str) -> list:
    return []
