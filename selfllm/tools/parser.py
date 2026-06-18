"""Parse tool calls from model output text."""

import json
import re
from typing import Any, Dict, List, Optional


class FunctionCallParser:
    """Parse function calls from model-generated text."""

    FORMAT_XML = "xml"
    FORMAT_JSON = "json"
    FORMAT_TAG = "tag"

    @classmethod
    def parse(cls, text: str, format: str = "xml") -> Optional[Dict[str, Any]]:
        """
        Extract a tool call from model output.

        Args:
            text: Model-generated text to parse.
            format: Format to parse ("xml", "json", or "tag").

        Returns:
            {"name": str, "arguments": Dict} or None if no tool call found.
        """
        if format == "xml":
            return cls._parse_xml(text)
        elif format == "json":
            return cls._parse_json(text)
        elif format == "tag":
            return cls._parse_tag(text)
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def _parse_xml(cls, text: str) -> Optional[Dict[str, Any]]:
        """Parse <tool>name</tool><args>{...}</args> format."""
        tool_match = re.search(r"<tool>(\w+)</tool>", text)
        if not tool_match:
            return None

        name = tool_match.group(1)

        args_match = re.search(r"<args>(\{.*?\})</args>", text, re.DOTALL)
        if args_match:
            try:
                arguments = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = {}

        return {"name": name, "arguments": arguments}

    @classmethod
    def _parse_json(cls, text: str) -> Optional[Dict[str, Any]]:
        """Parse {"name": "...", "arguments": {...}} format."""
        try:
            # Look for JSON object with "name" and "arguments" keys
            match = re.search(
                r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}',
                text,
                re.DOTALL,
            )
            if match:
                name = match.group(1)
                arguments = json.loads(match.group(2))
                return {"name": name, "arguments": arguments}
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    @classmethod
    def _parse_tag(cls, text: str) -> Optional[Dict[str, Any]]:
        """Parse <function=name>{...}</function> format."""
        match = re.search(r"<function=(\w+)>(.*?)</function>", text, re.DOTALL)
        if match:
            name = match.group(1)
            try:
                arguments = json.loads(match.group(2))
            except json.JSONDecodeError:
                arguments = {}
            return {"name": name, "arguments": arguments}
        return None

    @classmethod
    def inject_tool_prompt(
        cls, system_prompt: str, tools: List[Dict[str, Any]]
    ) -> str:
        """
        Add tool definitions to the system prompt.

        This teaches the model what tools are available and how to call them.

        Args:
            system_prompt: Original system prompt.
            tools: List of tool definitions (from ToolRegistry.list_tools()).

        Returns:
            Augmented system prompt with tool definitions.
        """
        tool_desc = (
            "\n\nYou have access to the following tools. When you need to use a tool, "
            "format your response as:\n"
        )
        tool_desc += '<tool>TOOL_NAME</tool><args>{"param": "value"}</args>\n\n'
        tool_desc += "Available tools:\n"

        for tool in tools:
            tool_desc += f"\n- {tool['name']}: {tool['description']}\n"
            if "parameters" in tool and "properties" in tool["parameters"]:
                for param_name, param_info in tool["parameters"]["properties"].items():
                    desc = param_info.get("description", "No description")
                    tool_desc += f"  - {param_name}: {desc}\n"

        tool_desc += (
            "\nAfter using a tool, you will see the result. "
            "Continue until you have a complete answer.\n"
        )
        tool_desc += (
            "When you have the final answer, respond directly "
            "without using any tool tags.\n"
        )

        return system_prompt + tool_desc
