"""
registry.py - Tool discovery and per-run capability manifest contracts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class ToolDescriptor:
    """
    Human- and machine-readable description of one callable tool.

    The descriptor is stable across transport backends. The worker and planner
    reason over these fields, while the underlying execution may still route
    through n8n or a future native connector.
    """

    name: str
    description: str = ""
    actions: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=lambda: {"type": "object"})
    risk: str = "unknown"
    read_only: bool = False
    requires_approval: bool = False
    backend: str = "webhook"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ToolDescriptor":
        actions = payload.get("actions", [])
        if isinstance(actions, str):
            actions = [actions]
        return cls(
            name=str(payload.get("name", "") or "").strip(),
            description=str(payload.get("description", "") or ""),
            actions=[str(action).strip() for action in actions if str(action).strip()],
            input_schema=payload.get("input_schema")
            if isinstance(payload.get("input_schema"), dict)
            else {"type": "object"},
            risk=str(payload.get("risk", "unknown") or "unknown"),
            read_only=bool(payload.get("read_only", False)),
            requires_approval=bool(payload.get("requires_approval", False)),
            backend=str(payload.get("backend", "webhook") or "webhook"),
            metadata=payload.get("metadata")
            if isinstance(payload.get("metadata"), dict)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def supports_action(self, action: str) -> bool:
        if not self.actions:
            return True
        return action in self.actions


@dataclass(slots=True)
class CapabilityManifest:
    """
    Per-run authorization manifest describing which tools the agent may use.
    """

    run_id: str = ""
    user_id: str = ""
    tools: list[ToolDescriptor] = field(default_factory=list)
    manifest_version: str = "1"
    generated_at: str = field(default_factory=_utc_now)
    source: str = "runtime"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CapabilityManifest":
        raw_tools = payload.get("tools", [])
        tools: list[ToolDescriptor] = []
        if isinstance(raw_tools, list):
            for item in raw_tools:
                if isinstance(item, ToolDescriptor):
                    tools.append(item)
                elif isinstance(item, dict):
                    descriptor = ToolDescriptor.from_dict(item)
                    if descriptor.name:
                        tools.append(descriptor)
                elif isinstance(item, str) and item.strip():
                    tools.append(ToolDescriptor(name=item.strip()))
        return cls(
            run_id=str(payload.get("run_id", "") or ""),
            user_id=str(payload.get("user_id", "") or ""),
            tools=tools,
            manifest_version=str(payload.get("manifest_version", "1") or "1"),
            generated_at=str(payload.get("generated_at") or _utc_now()),
            source=str(payload.get("source", "runtime") or "runtime"),
            metadata=payload.get("metadata")
            if isinstance(payload.get("metadata"), dict)
            else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "user_id": self.user_id,
            "tools": [tool.to_dict() for tool in self.tools],
            "manifest_version": self.manifest_version,
            "generated_at": self.generated_at,
            "source": self.source,
            "metadata": dict(self.metadata),
        }

    def tool_names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def get_tool(self, name: str) -> ToolDescriptor | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def allows(self, tool_name: str, action: str) -> bool:
        tool = self.get_tool(tool_name)
        if tool is None:
            return False
        return tool.supports_action(action)

    def bind(self, *, run_id: str, user_id: str) -> "CapabilityManifest":
        return CapabilityManifest(
            run_id=run_id,
            user_id=user_id,
            tools=[ToolDescriptor.from_dict(tool.to_dict()) for tool in self.tools],
            manifest_version=self.manifest_version,
            generated_at=self.generated_at,
            source=self.source,
            metadata=dict(self.metadata),
        )


class ToolRegistry:
    """
    Runtime registry of tool descriptors known to the current worker process.

    The registry answers "what connectors exist here?" while the capability
    manifest answers "which subset is allowed for this run?".
    """

    def __init__(self, tools: Iterable[ToolDescriptor | dict[str, Any]] | None = None) -> None:
        self._tools: dict[str, ToolDescriptor] = {}
        if tools is not None:
            self.register_many(tools)

    @classmethod
    def from_manifest(cls, manifest: CapabilityManifest | dict[str, Any]) -> "ToolRegistry":
        resolved = (
            manifest
            if isinstance(manifest, CapabilityManifest)
            else CapabilityManifest.from_dict(manifest)
        )
        return cls(resolved.tools)

    def register(self, descriptor: ToolDescriptor | dict[str, Any]) -> ToolDescriptor:
        resolved = (
            descriptor
            if isinstance(descriptor, ToolDescriptor)
            else ToolDescriptor.from_dict(descriptor)
        )
        if not resolved.name:
            raise ValueError("ToolDescriptor.name is required.")
        self._tools[resolved.name] = resolved
        return resolved

    def register_many(self, tools: Iterable[ToolDescriptor | dict[str, Any]]) -> None:
        for descriptor in tools:
            self.register(descriptor)

    def get(self, name: str) -> ToolDescriptor | None:
        return self._tools.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[ToolDescriptor]:
        return list(self._tools.values())

    def to_manifest(
        self,
        *,
        run_id: str = "",
        user_id: str = "",
        allowed_tools: Iterable[str] | None = None,
        source: str = "registry",
        metadata: dict[str, Any] | None = None,
    ) -> CapabilityManifest:
        if allowed_tools is None:
            selected = self.list_tools()
        else:
            selected = [
                descriptor
                for name in allowed_tools
                if (descriptor := self.get(name)) is not None
            ]
        return CapabilityManifest(
            run_id=run_id,
            user_id=user_id,
            tools=[ToolDescriptor.from_dict(tool.to_dict()) for tool in selected],
            source=source,
            metadata=metadata or {},
        )


TOOL_DESCRIPTOR_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ToolDescriptor",
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "actions", "input_schema", "risk", "read_only"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "actions": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "input_schema": {"type": "object"},
        "risk": {"type": "string", "minLength": 1},
        "read_only": {"type": "boolean"},
        "requires_approval": {"type": "boolean"},
        "backend": {"type": "string"},
        "metadata": {"type": "object"},
    },
}


CAPABILITY_MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "CapabilityManifest",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "run_id",
        "user_id",
        "tools",
        "manifest_version",
        "generated_at",
        "source",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "user_id": {"type": "string"},
        "tools": {
            "type": "array",
            "items": TOOL_DESCRIPTOR_SCHEMA,
        },
        "manifest_version": {"type": "string", "const": "1"},
        "generated_at": {"type": "string", "format": "date-time"},
        "source": {"type": "string", "minLength": 1},
        "metadata": {"type": "object"},
    },
}
