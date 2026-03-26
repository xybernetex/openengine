"""
contracts.py - Stable tool/connector contracts for world interaction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ToolInvocation:
    """
    Canonical tool request envelope emitted by the agent core.

    The transport backend may be n8n today and native connectors later, but
    this payload must remain stable across implementations.
    """

    invocation_id: str
    run_id: str
    step_id: str
    user_id: str
    tool_name: str
    action: str
    input: dict[str, Any] = field(default_factory=dict)
    requested_at: str = field(default_factory=_utc_now)
    contract_version: str = "1"

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        step_id: str,
        user_id: str,
        tool_name: str,
        action: str,
        input: dict[str, Any] | None = None,
        invocation_id: str | None = None,
    ) -> "ToolInvocation":
        return cls(
            invocation_id=invocation_id or uuid4().hex,
            run_id=run_id,
            step_id=step_id,
            user_id=user_id,
            tool_name=tool_name,
            action=action,
            input=input or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    """
    Canonical connector response envelope returned to the agent core.
    """

    invocation_id: str
    run_id: str
    status: str
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    completed_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    @classmethod
    def failure(
        cls,
        invocation: ToolInvocation,
        error: str,
        *,
        status: str = "failed",
        output: dict[str, Any] | None = None,
    ) -> "ToolResult":
        return cls(
            invocation_id=invocation.invocation_id,
            run_id=invocation.run_id,
            status=status,
            output=output or {},
            error=error,
        )


class ToolGateway(ABC):
    """
    Stable execution interface for external world actions.

    Implementations may be backed by n8n webhooks, internal services, or
    native connectors, but the worker integrates only against this interface.
    """

    @abstractmethod
    def invoke(self, invocation: ToolInvocation) -> ToolResult:
        """
        Execute one tool request and return a normalized result.
        """


TOOL_INVOCATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ToolInvocation",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "invocation_id",
        "run_id",
        "step_id",
        "user_id",
        "tool_name",
        "action",
        "input",
        "requested_at",
        "contract_version",
    ],
    "properties": {
        "invocation_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "step_id": {"type": "string", "minLength": 1},
        "user_id": {"type": "string"},
        "tool_name": {"type": "string", "minLength": 1},
        "action": {"type": "string", "minLength": 1},
        "input": {"type": "object"},
        "requested_at": {"type": "string", "format": "date-time"},
        "contract_version": {"type": "string", "const": "1"},
    },
}


TOOL_RESULT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ToolResult",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "invocation_id",
        "run_id",
        "status",
        "output",
        "error",
        "completed_at",
    ],
    "properties": {
        "invocation_id": {"type": "string", "minLength": 1},
        "run_id": {"type": "string", "minLength": 1},
        "status": {
            "type": "string",
            "enum": ["succeeded", "failed", "timed_out"],
        },
        "output": {"type": "object"},
        "error": {"type": "string"},
        "completed_at": {"type": "string", "format": "date-time"},
    },
}


TOOL_CALL_EVENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ToolCallEvent",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "data"],
    "properties": {
        "type": {"type": "string", "const": "tool_call"},
        "data": {
            "type": "object",
            "additionalProperties": False,
            "required": ["invocation_id", "tool_name", "action", "status"],
            "properties": {
                "invocation_id": {"type": "string", "minLength": 1},
                "tool_name": {"type": "string", "minLength": 1},
                "action": {"type": "string", "minLength": 1},
                "status": {"type": "string", "const": "started"},
            },
        },
    },
}


TOOL_RESULT_EVENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ToolResultEvent",
    "type": "object",
    "additionalProperties": False,
    "required": ["type", "data"],
    "properties": {
        "type": {"type": "string", "const": "tool_result"},
        "data": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "invocation_id",
                "tool_name",
                "status",
                "output",
                "error",
            ],
            "properties": {
                "invocation_id": {"type": "string", "minLength": 1},
                "tool_name": {"type": "string", "minLength": 1},
                "status": {
                    "type": "string",
                    "enum": ["succeeded", "failed", "timed_out"],
                },
                "output": {"type": "object"},
                "error": {"type": "string"},
            },
        },
    },
}
