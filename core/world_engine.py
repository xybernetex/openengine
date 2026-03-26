import os
import json
import logging
import platform
from typing import Optional, Dict, Any, List

# Core dependency for stochastic JSON recovery
from json_repair import repair_json

class Archivist:
    """
    State Persistence Manager. 
    Handles the serialization of environment events and artifact history.
    """
    def __init__(self, wm: Any):
        self.wm = wm
        self.logger = logging.getLogger("Archivist")

    def archive_artifact(self, run_id: str, artifact_type: str, content: str, aspect_id: str) -> str:
        """Saves a generated artifact to the working memory."""
        return self.wm.add_artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            content=content,
            parent_aspect_id=aspect_id
        )

class Integrator:
    """
    LLM Orchestration Layer.
    Translates RL actions into LLM prompts and handles the response logic.
    """
    def __init__(self, cf_client: Any, model: str = "@cf/meta/llama-3-8b-instruct"):
        self.cf = cf_client
        self.model = model
        self.host_os = platform.system()
        self.logger = logging.getLogger("Integrator")

    def _get_system_context(self, goal: str) -> str:
        """Injects Host OS awareness into the system prompt."""
        os_constraint = (
            f"CRITICAL: The current execution environment is {self.host_os}. "
            "All executable artifacts MUST use native syntax: "
        )
        if self.host_os == "Windows":
            os_constraint += "PowerShell (.ps1) or Python (.py). DO NOT generate Bash (.sh)."
        else:
            os_constraint += "Bash (.sh) or Python (.py)."

        return (
            f"You are the Xybernetex Integrator. {os_constraint}\n"
            f"Objective: {goal}\n"
            "Return only valid JSON with 'content', 'type', and 'explanation'."
        )

    def synthesize(self, goal: str, artifact_type: str, aspect_id: str) -> Dict[str, Any]:
        """Calls the LLM to generate the artifact based on host constraints."""
        system_prompt = self._get_system_context(goal)
        user_prompt = f"Generate a {artifact_type} for aspect: {aspect_id}"

        raw_response = self.cf.complete(
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        try:
            return json.loads(repair_json(raw_response))
        except Exception as e:
            self.logger.error(f"Integrator failed to parse LLM output: {e}")
            raise

class WorldEngine:
    """
    High-level Manager for the Xybernetex Generative Core.
    """
    def __init__(self, wm: Any, cf_client: Any):
        self.archivist = Archivist(wm)
        self.integrator = Integrator(cf_client)

    def generate_artifact(self, run_id: str, goal: str, aspect_id: str, artifact_type: str) -> str:
        """Coordinates the generation and archiving of a new artifact."""
        result = self.integrator.synthesize(goal, artifact_type, aspect_id)
        return self.archivist.archive_artifact(
            run_id, 
            artifact_type, 
            result.get("content", ""), 
            aspect_id
        )