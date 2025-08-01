# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
from ageval.evaluators.agent.tools.base import ToolBase

_TOOL_REGISTRY: dict[str, type[ToolBase]] = {}


def register_tool(name):
    def decorate(fn):
        assert (
            name not in _TOOL_REGISTRY
        ), f"tool named '{name}' conflicts with existing registered tool!"

        _TOOL_REGISTRY[name] = fn
        return fn

    return decorate


def get_tools(*, enabled_tools: set[str] | None = None) -> dict[str, type[ToolBase]]:
    if enabled_tools is None:
        return _TOOL_REGISTRY.copy()

    return {k: v for k, v in _TOOL_REGISTRY.items() if k in enabled_tools}
