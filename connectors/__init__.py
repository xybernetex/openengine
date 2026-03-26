"""
Connector runtime contracts for external world interaction.
"""

from connectors.contracts import (
    TOOL_CALL_EVENT_SCHEMA,
    TOOL_INVOCATION_SCHEMA,
    TOOL_RESULT_EVENT_SCHEMA,
    TOOL_RESULT_SCHEMA,
    ToolGateway,
    ToolInvocation,
    ToolResult,
)
from connectors.n8n_webhook import N8NWebhookConnector
from connectors.registry import (
    CAPABILITY_MANIFEST_SCHEMA,
    TOOL_DESCRIPTOR_SCHEMA,
    CapabilityManifest,
    ToolDescriptor,
    ToolRegistry,
)

__all__ = [
    "CAPABILITY_MANIFEST_SCHEMA",
    "N8NWebhookConnector",
    "CapabilityManifest",
    "TOOL_CALL_EVENT_SCHEMA",
    "TOOL_DESCRIPTOR_SCHEMA",
    "TOOL_INVOCATION_SCHEMA",
    "TOOL_RESULT_EVENT_SCHEMA",
    "TOOL_RESULT_SCHEMA",
    "ToolGateway",
    "ToolDescriptor",
    "ToolInvocation",
    "ToolRegistry",
    "ToolResult",
]
