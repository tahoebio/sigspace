"""Utility functions for Tahoe Agent."""

import json
from typing import Any, Dict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def pretty_print(message: BaseMessage) -> tuple[str, str]:
    """Pretty print a message for logging purposes.

    Args:
        message: The message to print

    Returns:
        Tuple of (role, content) for logging
    """
    if isinstance(message, HumanMessage):
        role = "user"
        content = message.content
    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message.content
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls_str = "\n".join(
                [f"Tool call: {tc['name']}({tc['args']})" for tc in message.tool_calls]
            )
            content = f"{content}\n{tool_calls_str}" if content else tool_calls_str
    elif isinstance(message, SystemMessage):
        role = "system"
        content = message.content
    elif isinstance(message, ToolMessage):
        role = "tool"
        content = f"Tool: {message.name}\nResult: {message.content}"
    else:
        role = "unknown"
        content = str(message.content)

    return role, content


def format_tool_schema(tool_schema: Dict[str, Any]) -> str:
    """Format a tool schema for display in prompts.

    Args:
        tool_schema: Dictionary containing tool information

    Returns:
        Formatted string representation of the tool
    """
    name = tool_schema.get("name", "Unknown")
    description = tool_schema.get("description", "No description")
    parameters = tool_schema.get("parameters", {})

    formatted = f"**{name}**: {description}"

    if parameters and isinstance(parameters, dict):
        if "properties" in parameters:
            props = parameters["properties"]
            required = parameters.get("required", [])

            param_list = []
            for param_name, param_info in props.items():
                param_type = param_info.get("type", "unknown")
                param_desc = param_info.get("description", "")
                is_required = param_name in required
                req_str = " (required)" if is_required else " (optional)"
                param_list.append(
                    f"  - {param_name} ({param_type}){req_str}: {param_desc}"
                )

            if param_list:
                formatted += "\n  Parameters:\n" + "\n".join(param_list)

    return formatted


def safe_json_loads(text: str) -> Any:
    """Safely load JSON from text, returning None if parsing fails.

    Args:
        text: Text to parse as JSON

    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def extract_code_blocks(text: str, language: str = "") -> list[str]:
    """Extract code blocks from markdown text.

    Args:
        text: Text containing markdown code blocks
        language: Specific language to filter for (e.g., "python")

    Returns:
        List of code block contents
    """
    import re

    if language:
        pattern = rf"```{language}\n(.*?)\n```"
    else:
        pattern = r"```(?:\w+)?\n(.*?)\n```"

    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]
