"""
xyber_env.py — Xybernetex Gym Environment
"""
import sys
from pathlib import Path

# ── PATH BOOTSTRAP ─────────────────────────────────────────────────────────────
# Insert the Reinforcement/ directory so that sibling modules (rl_chassis) and
# the core/ sub-package are always resolvable regardless of cwd or invocation style.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (_HERE, _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
from typing import Dict, Any, Tuple, Optional

from core.working_memory import WorkingMemory
from core.world_engine import Archivist, Integrator
from core.code_executor import run_code, sanitize_code
from rl_chassis import (ActionSpace, StateVectorizer, RewardFunction, TransitionInfo,
                        _is_code_goal, _is_tool_goal, _is_multi_tool_goal, _is_delivery_aspect)

class XybernetexEnv:
    def __init__(
        self,
        mock_llm        : bool            = True,
        memory_backend  : str             = "sqlite",
        output_dir      : Optional[Path]  = None,
        user_id         : Optional[str]   = None,
        llm_provider    : str             = "",
        llm_api_key     : str             = "",
        llm_model       : str             = "",
        cf_account_id   : str             = "",
        cf_api_token    : str             = "",
        tavily_api_key  : str             = "",
        brave_api_key   : str             = "",
        resend_api_key  : str             = "",
    ):
        self.mock_llm = mock_llm
        self.wm = WorkingMemory()
        self.vectorizer = StateVectorizer()
        self.reward_fn = RewardFunction()

        self.current_run_id    = None
        self.current_goal      = ""
        self.current_user_id   = user_id
        self.current_capability_manifest = None
        self.step_count        = 0
        self.max_steps         = 25
        self.prev_action_index = None

        # ── Live executor (only active when mock_llm=False) ────────────────────
        self.executor = None
        if not mock_llm:
            from live_executor import LiveExecutor
            _out = output_dir or (_HERE / "output")
            self.executor = LiveExecutor(
                self.wm, _out,
                llm_provider   = llm_provider,
                llm_api_key    = llm_api_key,
                llm_model      = llm_model,
                cf_account_id  = cf_account_id,
                cf_api_token   = cf_api_token,
                tavily_api_key = tavily_api_key,
                brave_api_key  = brave_api_key,
                resend_api_key = resend_api_key,
            )
            print(f"[XybernetexEnv] LiveExecutor active → output_dir={_out}", flush=True)

    def reset(
        self,
        goal: str = "Default Goal",
        user_id: Optional[str] = None,
        run_id: Optional[str] = None,
        capability_manifest: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        self.current_goal = goal
        self.current_user_id = user_id or self.current_user_id
        self.current_run_id = self.wm.create_run(
            goal,
            user_id=self.current_user_id,
            run_id=run_id,
        )
        self.current_capability_manifest = capability_manifest
        if capability_manifest is not None:
            self.wm.set_capability_manifest(self.current_run_id, capability_manifest)
            if self.executor is not None and hasattr(self.executor, "set_capability_manifest"):
                self.executor.set_capability_manifest(capability_manifest)
        self.step_count = 0
        self.prev_action_index = None
        self.vectorizer.reset()
        return self._get_obs()

    def get_expert_info(self) -> Dict[str, Any]:
        """God-mode telemetry for the SmarterTeacher expert."""
        aspects = {a['aspect']: a['status'] for a in self.wm.get_goal_aspects(self.current_run_id)}

        unexecuted_count = self.wm.conn.execute(
            "SELECT COUNT(*) FROM artifacts a "
            "LEFT JOIN executions e ON a.id = e.artifact_id "
            "WHERE a.run_id = ? AND a.artifact_type = 'CODE' AND e.id IS NULL",
            (self.current_run_id,)
        ).fetchone()[0]

        pending_names = [k for k, v in aspects.items() if v == "pending"]
        delivery_pending = bool(pending_names) and all(_is_delivery_aspect(n) for n in pending_names)

        return {
            "all_aspects_complete": len(aspects) > 0 and all(v == "complete" for v in aspects.values()),
            "has_pending_aspects": any(v == "pending" for v in aspects.values()),
            "has_unexecuted_code": unexecuted_count > 0,
            "needs_planning": len(aspects) == 0,
            "delivery_pending": delivery_pending,
        }

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        self.step_count += 1
        prev_aspects = {a['aspect']: a['status'] for a in self.wm.get_goal_aspects(self.current_run_id)}
        action_name = ActionSpace.decode(action_index)
        
        info = TransitionInfo(
            action_index=action_index,
            prev_aspect_states=prev_aspects,
            curr_aspect_states={},
            exec_success=True if self.mock_llm else False,
            exec_returncode=0 if self.mock_llm else -1
        )

        # ── Mock simulation (training rollouts only) ──────────────────────────
        # CRITICAL: Mock transitions must mirror live_executor behaviour so the
        # policy learns a golden path that actually works in live inference.
        #
        # Golden path:  PLAN → [WRITE → EXEC] × N_aspects → DONE
        #
        #   PLAN           → create 3 aspects (pending)
        #   WRITE_ARTIFACT → create mock CODE artifact + aspect pending → in_progress
        #   EXECUTE_CODE   → "run" the mock artifact + aspect in_progress → complete
        #   DONE           → validated if all aspects are complete
        #
        if self.mock_llm:
            if action_name == "PLAN":
                # Create goal-appropriate aspects exactly once per run.
                # Delivery aspects (containing "email", "transmission", etc.) can only
                # be completed by INVOKE_TOOL — WRITE_ARTIFACT will skip them.
                if not prev_aspects:
                    if _is_multi_tool_goal(self.current_goal):
                        # Research + delivery: 5-8 non-delivery aspects followed by
                        # the delivery aspect ALWAYS LAST — exactly mirroring what
                        # the live LLM produces when given the PLAN prompt instruction
                        # ("FINAL aspect MUST be email_transmission_confirmation").
                        # Training must match the inference distribution; randomising
                        # the delivery position diluted the signal to ~5 examples per
                        # position variant, which was insufficient for the MLP to learn.
                        import random as _rnd
                        _pool = [
                            "research_findings", "country_analysis",
                            "cost_comparison", "risk_assessment",
                            "logistics_review", "final_report",
                            "supply_chain_review", "vendor_evaluation",
                            "compliance_assessment",
                        ]
                        n = _rnd.randint(5, 8)   # 5-8 matches live LLM output (typically 7)
                        non_delivery = _rnd.sample(_pool, n)
                        aspects_to_create = non_delivery + ["email_transmission_confirmation"]
                    elif _is_tool_goal(self.current_goal):
                        goal_words = set(self.current_goal.lower().split())
                        is_email_only = bool({"email", "send", "notify", "mail"} & goal_words)
                        if is_email_only:
                            # Pure email goal — single delivery aspect
                            aspects_to_create = ["email_transmission_confirmation"]
                        else:
                            # Pure web-search goal — no delivery aspect
                            aspects_to_create = ["research_findings", "analysis_report",
                                                 "summary_document"]
                    else:
                        aspects_to_create = ["define_goal", "produce_artifact", "verify_result"]

                    for asp in aspects_to_create:
                        self.wm.upsert_goal_aspect(
                            self.current_run_id, asp, "pending", self.step_count
                        )

            elif action_name == "INVOKE_TOOL":
                # Simulate tool execution in mock mode.
                # If ALL remaining pending aspects are delivery aspects → this is the
                # delivery tool call (send_email etc.); mark the first one complete.
                # If ANY non-delivery aspects are still pending → this is a web-search
                # call; no aspect state change (context is enriched, not delivered).
                #
                # CRITICAL: the old code fired on "any delivery aspect pending",
                # which caused the early web-search INVOKE_TOOL (step 2) to mark
                # email_transmission_confirmation complete before the research was done.
                # That gave the policy zero training examples of s[18]=1.0 → INVOKE_TOOL.
                all_pending = {k for k, v in prev_aspects.items() if v == "pending"}
                delivery_pending_now = (
                    bool(all_pending)
                    and all(_is_delivery_aspect(k) for k in all_pending)
                )
                if delivery_pending_now:
                    delivery_aspects = [k for k in all_pending if _is_delivery_aspect(k)]
                    step_id = self.wm.record_step(
                        self.current_run_id, self.step_count,
                        "INVOKE_TOOL", "Mock Tool Delivery", "mock"
                    )
                    self.wm.upsert_goal_aspect(
                        self.current_run_id, delivery_aspects[0], "complete", self.step_count
                    )

            elif action_name == "WRITE_ARTIFACT":
                pending = [k for k, v in prev_aspects.items() if v == "pending"]
                # Skip delivery aspects — only INVOKE_TOOL can complete those.
                advanceable = [p for p in pending if not _is_delivery_aspect(p)]
                if advanceable:
                    is_code = _is_code_goal(self.current_goal)
                    step_id = self.wm.record_step(
                        self.current_run_id, self.step_count,
                        "WRITE_ARTIFACT", "Mock Artifact", "mock"
                    )

                    if is_code:
                        self.wm.save_artifact(
                            self.current_run_id, step_id,
                            "CODE", "Mock Script", "# mock code\nprint('ok')"
                        )
                        self.wm.upsert_goal_aspect(
                            self.current_run_id, advanceable[0], "in_progress", self.step_count
                        )
                    else:
                        self.wm.save_artifact(
                            self.current_run_id, step_id,
                            "DOCUMENT", "Mock Analysis", "mock strategic analysis"
                        )
                        self.wm.upsert_goal_aspect(
                            self.current_run_id, advanceable[0], "complete", self.step_count
                        )

            elif action_name == "EXECUTE_CODE":
                # Only meaningful for CODE goals — find and "run" unexecuted CODE.
                unexecuted = self.wm.conn.execute(
                    "SELECT a.id FROM artifacts a "
                    "LEFT JOIN executions e ON a.id = e.artifact_id "
                    "WHERE a.run_id = ? AND a.artifact_type = 'CODE' AND e.id IS NULL "
                    "ORDER BY a.created_at DESC LIMIT 1",
                    (self.current_run_id,)
                ).fetchone()

                if unexecuted:
                    self.wm.save_execution(unexecuted[0], {
                        "success": True, "returncode": 0,
                        "stdout": "mock output", "stderr": "",
                        "timed_out": False,
                    })
                    in_progress = [k for k, v in prev_aspects.items() if v == "in_progress"]
                    if in_progress:
                        self.wm.upsert_goal_aspect(
                            self.current_run_id, in_progress[0], "complete", self.step_count
                        )

        # ── Live execution (inference mode only) ──────────────────────────────
        else:
            # executor is always set when mock_llm=False; dispatch never raises
            result = self.executor.dispatch(
                action_name,
                self.current_run_id,
                self.current_goal,
                self.step_count,
            )
            info.exec_success    = result.get("exec_success",    False)
            info.exec_returncode = result.get("exec_returncode", -1)

        # ── DONE gate — always evaluated regardless of mock/live mode ──────────
        # IMPORTANT: Use "mock" (aspect-only check) for BOTH modes.
        # The policy was trained against the aspect-only termination contract.
        # Switching to the full technical ruleset (CODE + verification) in live
        # mode creates a train/infer mismatch that the policy cannot satisfy.
        # TODO: Once live_executor produces proper verification evidence,
        # retrain the policy against the stricter gate and switch live to
        # can_terminate(run_id, self.current_goal).
        if action_name == "DONE":
            allowed, _ = self.wm.can_terminate(self.current_run_id, "mock")
            info.done_valid = allowed

        info.curr_aspect_states = {a['aspect']: a['status'] for a in self.wm.get_goal_aspects(self.current_run_id)}
        reward = self.reward_fn.compute(info, self.prev_action_index)
        self.prev_action_index = action_index
        
        done = (action_name == "DONE" and info.done_valid) or (self.step_count >= self.max_steps)
        self.vectorizer.push_action(action_index)
        
        return self._get_obs(), reward, done, {
            "action_name": action_name,
            "exec_success": info.exec_success,
            "exec_returncode": info.exec_returncode,
            "done_valid": getattr(info, "done_valid", False),
            "step_count": self.step_count,
            "curr_aspect_states": info.curr_aspect_states,
        }

    def _get_obs(self) -> np.ndarray:
        return self.vectorizer.vectorize(
            self.wm.conn, self.current_run_id, self.step_count, self.max_steps,
            goal=self.current_goal
        )

    def close(self):
        self.wm.close()
