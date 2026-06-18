"""Tool registry and execution engine for SelfLLM."""

import ast
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class Tool:
    """A tool that the model can call."""

    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    function: Callable
    timeout: float = 30.0


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_builtins()

    def _register_builtins(self):
        """Register all built-in tools."""
        self.register(
            Tool(
                name="calculator",
                description="Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses, math functions.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression to evaluate, e.g. '15 * 23' or 'sqrt(16) + 5'",
                        }
                    },
                    "required": ["expression"],
                },
                function=self._calculator,
            )
        )

        self.register(
            Tool(
                name="python",
                description="Execute Python code in a sandboxed environment. Use for complex calculations, data processing, or algorithm implementation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"],
                },
                function=self._python,
            )
        )

        self.register(
            Tool(
                name="file_read",
                description="Read the contents of a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to the file"}
                    },
                    "required": ["path"],
                },
                function=self._file_read,
            )
        )

        self.register(
            Tool(
                name="file_write",
                description="Write content to a file.",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Path to write"},
                        "content": {
                            "type": "string",
                            "description": "Content to write",
                        },
                    },
                    "required": ["path", "content"],
                },
                function=self._file_write,
            )
        )

        self.register(
            Tool(
                name="http_get",
                description="Fetch content from a URL via HTTP GET.",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to fetch"}
                    },
                    "required": ["url"],
                },
                function=self._http_get,
            )
        )

        self.register(
            Tool(
                name="search",
                description="Search for information. (Placeholder - returns query echo)",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
                function=self._search,
            )
        )

        self.register(
            Tool(
                name="current_time",
                description="Get the current date and time.",
                parameters={"type": "object", "properties": {}, "required": []},
                function=self._current_time,
            )
        )

        self.register(
            Tool(
                name="count_words",
                description="Count words and characters in text.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to analyze"}
                    },
                    "required": ["text"],
                },
                function=self._count_words,
            )
        )

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(
                f"Tool '{name}' not found. Available: {list(self._tools.keys())}"
            )
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """List all registered tools with their schemas."""
        return [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in self._tools.values()
        ]

    def get_schemas(self) -> str:
        """Get tool schemas formatted for prompt injection."""
        lines = ["Available tools:"]
        for t in self._tools.values():
            lines.append(f"\n### {t.name}")
            lines.append(f"Description: {t.description}")
            lines.append(f"Parameters: {json.dumps(t.parameters, indent=2)}")
            lines.append(f"Usage: <tool>{t.name}</tool><args>{{json args}}</args>")
        return "\n".join(lines)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # --- Built-in tool implementations ---

    @staticmethod
    def _calculator(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        try:
            # Whitelist of allowed names
            allowed_names = {
                "abs": abs,
                "round": round,
                "max": max,
                "min": min,
                "sum": sum,
                "len": len,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "floor": math.floor,
                "ceil": math.ceil,
                "pi": math.pi,
                "e": math.e,
            }
            # Parse and evaluate
            tree = ast.parse(expression.strip(), mode="eval")
            code = compile(tree, "<string>", "eval")
            result = eval(code, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _python(code: str) -> str:
        """Execute Python code in a sandbox."""
        try:
            stdout_buffer = []

            def _print(*args, **kwargs):
                stdout_buffer.append(" ".join(str(a) for a in args))

            safe_globals = {
                "__builtins__": {
                    "abs": abs,
                    "round": round,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "len": len,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "sorted": sorted,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "print": _print,
                    "Exception": Exception,
                    "True": True,
                    "False": False,
                    "None": None,
                },
                "math": math,
                "json": json,
                "re": re,
            }

            exec(code, safe_globals, {})
            return "\n".join(stdout_buffer) if stdout_buffer else "(no output)"
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"

    @staticmethod
    def _file_read(path: str) -> str:
        """Read the contents of a file."""
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _file_write(path: str, content: str) -> str:
        """Write content to a file."""
        try:
            dir_name = os.path.dirname(path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error: {str(e)}"

    @staticmethod
    def _http_get(url: str) -> str:
        """Fetch content from a URL via HTTP GET."""
        try:
            import urllib.request

            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (SelfLLM)"},
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode("utf-8", errors="ignore")[:5000]
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    @staticmethod
    def _search(query: str) -> str:
        """Search for information (placeholder)."""
        return f"Search results for '{query}': (Search tool is a placeholder. Integrate with a search API for real results.)"

    @staticmethod
    def _current_time() -> str:
        """Get the current date and time."""
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _count_words(text: str) -> str:
        """Count words and characters in text."""
        words = len(text.split())
        chars = len(text)
        return f"Words: {words}, Characters: {chars}"


class ToolExecutor:
    """Execute tool calls with error handling and timeouts."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def execute(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool call.

        Args:
            tool_call: {"name": str, "arguments": Dict}

        Returns:
            {"status": "success|error", "result": str, "error": str|None, "duration_ms": float}
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        start = time.time()
        try:
            if name not in self.registry:
                return {
                    "status": "error",
                    "result": "",
                    "error": f"Unknown tool: {name}",
                    "duration_ms": 0,
                }

            tool = self.registry.get(name)
            result = tool.function(**arguments)
            duration = (time.time() - start) * 1000

            return {
                "status": "success",
                "result": str(result),
                "error": None,
                "duration_ms": round(duration, 2),
            }
        except Exception as e:
            duration = (time.time() - start) * 1000
            return {
                "status": "error",
                "result": "",
                "error": f"{type(e).__name__}: {str(e)}",
                "duration_ms": round(duration, 2),
            }
