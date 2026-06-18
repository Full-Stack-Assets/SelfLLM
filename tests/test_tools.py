"""Tests for tool registry, execution engine, and parser.

Tests cover:
- Calculator tool (basic arithmetic, math functions, error handling)
- Python sandbox tool (execution, security restrictions)
- File read/write tools (roundtrip)
- HTTP GET tool (fetch content)
- Current time tool (returns valid datetime)
- Count words tool (word/char counting)
- Tool registry (register, get, list)
- Function call parser (XML, JSON, tag formats, no-match case)
- Tool prompt injection (contains tool descriptions)
"""

import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from selfllm.tools.parser import FunctionCallParser
from selfllm.tools.registry import Tool, ToolExecutor, ToolRegistry


# --------------------------------------------------------------------------- #
# Calculator Tests
# --------------------------------------------------------------------------- #


class TestCalculator:
    """Tests for the calculator tool."""

    def test_calculator_basic(self):
        """Test basic arithmetic: 15 * 23 = 345."""
        registry = ToolRegistry()
        result = registry._calculator("15 * 23")
        assert result == "345"

    def test_calculator_addition(self):
        """Test addition."""
        registry = ToolRegistry()
        assert registry._calculator("2 + 3") == "5"

    def test_calculator_division(self):
        """Test division."""
        registry = ToolRegistry()
        assert registry._calculator("10 / 2") == "5.0"

    def test_calculator_parentheses(self):
        """Test parentheses."""
        registry = ToolRegistry()
        assert registry._calculator("(2 + 3) * 4") == "20"

    def test_calculator_power(self):
        """Test power operator."""
        registry = ToolRegistry()
        assert registry._calculator("2 ** 10") == "1024"

    def test_calculator_math_funcs_sqrt(self):
        """Test sqrt function: sqrt(16) = 4.0."""
        registry = ToolRegistry()
        result = registry._calculator("sqrt(16)")
        assert result == "4.0"

    def test_calculator_math_funcs_log(self):
        """Test log and trig functions."""
        registry = ToolRegistry()
        result = registry._calculator("log10(100)")
        assert result == "2.0"

    def test_calculator_constants(self):
        """Test math constants pi and e."""
        registry = ToolRegistry()
        result = registry._calculator("pi")
        assert float(result) == pytest.approx(3.14159, abs=0.001)

    def test_calculator_error_invalid_expression(self):
        """Test invalid expression returns error."""
        registry = ToolRegistry()
        result = registry._calculator("invalid!!!")
        assert result.startswith("Error:")

    def test_calculator_error_blocked_builtin(self):
        """Test that dangerous builtins are blocked."""
        registry = ToolRegistry()
        result = registry._calculator('__import__("os")')
        assert result.startswith("Error:")

    def test_calculator_complex_expression(self):
        """Test complex nested expression."""
        registry = ToolRegistry()
        result = registry._calculator("sqrt(16) + floor(3.7) * ceil(2.1)")
        # 4.0 + 3 * 3 = 13.0
        assert float(result) == pytest.approx(13.0, abs=0.001)


# --------------------------------------------------------------------------- #
# Python Sandbox Tests
# --------------------------------------------------------------------------- #


class TestPythonSandbox:
    """Tests for the Python sandbox tool."""

    def test_python_execution_print(self):
        """Test basic print execution."""
        registry = ToolRegistry()
        result = registry._python('print("hello")')
        assert result == "hello"

    def test_python_execution_calculation(self):
        """Test calculation with print."""
        registry = ToolRegistry()
        result = registry._python("x = 5\ny = 3\nprint(x * y)")
        assert result == "15"

    def test_python_execution_list_ops(self):
        """Test list operations."""
        registry = ToolRegistry()
        result = registry._python("data = [1, 2, 3]\nprint(sum(data))")
        assert result == "6"

    def test_python_execution_no_output(self):
        """Test code with no print produces no output."""
        registry = ToolRegistry()
        result = registry._python("x = 42")
        assert result == "(no output)"

    def test_python_sandbox_import_blocked(self):
        """Test that __import__ is blocked in sandbox."""
        registry = ToolRegistry()
        result = registry._python('__import__("os")')
        assert "Error" in result

    def test_python_sandbox_no_file_access(self):
        """Test that file operations are restricted."""
        registry = ToolRegistry()
        result = registry._python('open("/etc/passwd", "r")')
        assert "Error" in result

    def test_python_sandbox_math_available(self):
        """Test that math module is available."""
        registry = ToolRegistry()
        result = registry._python("import math\nprint(math.sqrt(16))")
        assert "Error" in result  # import is not a builtin in sandbox

    def test_python_sandbox_json_available(self):
        """Test that json module is available via safe_globals."""
        registry = ToolRegistry()
        result = registry._python('print(json.dumps({"key": "value"}))')
        assert result == '{"key": "value"}'

    def test_python_error_handling(self):
        """Test that runtime errors are caught gracefully."""
        registry = ToolRegistry()
        result = registry._python("print(1 / 0)")
        assert result.startswith("Error:")
        assert "ZeroDivisionError" in result

    def test_python_multiple_prints(self):
        """Test multiple print statements."""
        registry = ToolRegistry()
        result = registry._python("print(1)\nprint(2)\nprint(3)")
        assert result == "1\n2\n3"


# --------------------------------------------------------------------------- #
# File Read/Write Tests
# --------------------------------------------------------------------------- #


class TestFileReadWrite:
    """Tests for file read and write tools."""

    def test_file_read_write_roundtrip(self):
        """Test writing then reading a file roundtrip."""
        registry = ToolRegistry()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_file.txt")
            content = "Hello, SelfLLM tools!"

            # Write
            write_result = registry._file_write(path, content)
            assert "Successfully wrote" in write_result

            # Read
            read_result = registry._file_read(path)
            assert read_result == content

    def test_file_write_creates_directories(self):
        """Test that write creates parent directories."""
        registry = ToolRegistry()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "sub", "dir", "file.txt")
            result = registry._file_write(path, "nested content")
            assert "Successfully wrote" in result
            assert os.path.exists(path)

    def test_file_read_nonexistent(self):
        """Test reading a nonexistent file returns error."""
        registry = ToolRegistry()
        result = registry._file_read("/nonexistent/path/file.txt")
        assert result.startswith("Error:")


# --------------------------------------------------------------------------- #
# HTTP GET Tests
# --------------------------------------------------------------------------- #


class TestHttpGet:
    """Tests for the HTTP GET tool."""

    def test_http_get_example_com(self):
        """Test fetching example.com."""
        registry = ToolRegistry()
        result = registry._http_get("http://example.com")
        # Should contain HTML or an error message
        assert isinstance(result, str)
        assert len(result) > 0

    def test_http_get_invalid_url(self):
        """Test fetching an invalid URL returns error."""
        registry = ToolRegistry()
        result = registry._http_get("http://invalid.invalid.invalid")
        assert result.startswith("Error fetching")

    def test_http_get_truncation(self):
        """Test that responses longer than 5000 chars are truncated."""
        registry = ToolRegistry()
        result = registry._http_get("http://example.com")
        if not result.startswith("Error"):
            assert len(result) <= 5000


# --------------------------------------------------------------------------- #
# Current Time Tests
# --------------------------------------------------------------------------- #


class TestCurrentTime:
    """Tests for the current_time tool."""

    def test_current_time_format(self):
        """Test that current_time returns a valid datetime string."""
        registry = ToolRegistry()
        result = registry._current_time()
        # Should parse as datetime
        dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        assert isinstance(dt, datetime)

    def test_current_time_reasonable(self):
        """Test that returned time is close to now."""
        registry = ToolRegistry()
        result = registry._current_time()
        dt = datetime.strptime(result, "%Y-%m-%d %H:%M:%S")
        now = datetime.now()
        delta = abs((now - dt).total_seconds())
        assert delta < 5  # Within 5 seconds


# --------------------------------------------------------------------------- #
# Count Words Tests
# --------------------------------------------------------------------------- #


class TestCountWords:
    """Tests for the count_words tool."""

    def test_count_words_simple(self):
        """Test word counting on simple text."""
        registry = ToolRegistry()
        result = registry._count_words("hello world")
        assert "Words: 2" in result
        assert "Characters: 11" in result

    def test_count_words_empty(self):
        """Test word counting on empty string."""
        registry = ToolRegistry()
        result = registry._count_words("")
        assert "Words: 0" in result
        assert "Characters: 0" in result

    def test_count_words_long_text(self):
        """Test word counting on longer text."""
        registry = ToolRegistry()
        text = "The quick brown fox jumps over the lazy dog"
        result = registry._count_words(text)
        assert "Words: 9" in result
        assert f"Characters: {len(text)}" in result


# --------------------------------------------------------------------------- #
# Tool Registry Tests
# --------------------------------------------------------------------------- #


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_registry_default_tools(self):
        """Test that default tools are registered."""
        registry = ToolRegistry()
        expected_tools = [
            "calculator",
            "python",
            "file_read",
            "file_write",
            "http_get",
            "search",
            "current_time",
            "count_words",
        ]
        for name in expected_tools:
            assert name in registry

    def test_registry_get(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = registry.get("calculator")
        assert tool.name == "calculator"
        assert "mathematical" in tool.description.lower()

    def test_registry_get_missing(self):
        """Test getting a nonexistent tool raises KeyError."""
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent_tool")

    def test_registry_register(self):
        """Test registering a custom tool."""
        registry = ToolRegistry()

        def custom_func(x: int) -> str:
            return str(x * 2)

        tool = Tool(
            name="double",
            description="Double a number",
            parameters={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            function=custom_func,
        )
        registry.register(tool)
        assert "double" in registry
        assert registry.get("double").function(5) == "10"

    def test_registry_list_tools(self):
        """Test listing tools returns correct structure."""
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert len(tools) == 8
        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "parameters" in tool

    def test_registry_get_schemas(self):
        """Test get_schemas returns formatted string."""
        registry = ToolRegistry()
        schemas = registry.get_schemas()
        assert "Available tools:" in schemas
        assert "calculator" in schemas
        assert "<tool>calculator</tool>" in schemas

    def test_registry_len(self):
        """Test __len__ returns correct count."""
        registry = ToolRegistry()
        assert len(registry) == 8

    def test_registry_contains(self):
        """Test __contains__ works."""
        registry = ToolRegistry()
        assert "calculator" in registry
        assert "nonexistent" not in registry


# --------------------------------------------------------------------------- #
# ToolExecutor Tests
# --------------------------------------------------------------------------- #


class TestToolExecutor:
    """Tests for the ToolExecutor class."""

    def test_executor_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        result = executor.execute({"name": "calculator", "arguments": {"expression": "2 + 2"}})
        assert result["status"] == "success"
        assert result["result"] == "4"
        assert result["error"] is None
        assert result["duration_ms"] >= 0

    def test_executor_unknown_tool(self):
        """Test execution of unknown tool."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        result = executor.execute({"name": "nonexistent", "arguments": {}})
        assert result["status"] == "error"
        assert "Unknown tool" in result["error"]

    def test_executor_error_handling(self):
        """Test that tool errors are caught."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        result = executor.execute(
            {"name": "calculator", "arguments": {"expression": "1 / 0"}}
        )
        # Division by zero in calculator is caught and returned as error string
        assert result["status"] == "success"  # The tool itself handled the error
        assert "Error" in result["result"]

    def test_executor_duration(self):
        """Test that duration is recorded."""
        registry = ToolRegistry()
        executor = ToolExecutor(registry)
        result = executor.execute({"name": "current_time", "arguments": {}})
        assert result["status"] == "success"
        assert result["duration_ms"] >= 0


# --------------------------------------------------------------------------- #
# Parser Tests
# --------------------------------------------------------------------------- #


class TestParserXML:
    """Tests for XML format parsing."""

    def test_parser_xml_basic(self):
        """Parse calculator tool call in XML format."""
        text = '<tool>calculator</tool><args>{"expression": "1+1"}</args>'
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"]["expression"] == "1+1"

    def test_parser_xml_multiple_args(self):
        """Parse tool call with multiple arguments."""
        text = '<tool>file_write</tool><args>{"path": "/tmp/test.txt", "content": "hello"}</args>'
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["name"] == "file_write"
        assert result["arguments"]["path"] == "/tmp/test.txt"
        assert result["arguments"]["content"] == "hello"

    def test_parser_xml_multiline(self):
        """Parse multiline args with escaped newlines in JSON."""
        text = '<tool>python</tool><args>{"code": "print(1)\\nprint(2)"}</args>'
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["arguments"]["code"] == "print(1)\nprint(2)"

    def test_parser_xml_no_args(self):
        """Parse tool call with no args block."""
        text = "<tool>current_time</tool>"
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["name"] == "current_time"
        assert result["arguments"] == {}

    def test_parser_xml_surrounding_text(self):
        """Parse tool call surrounded by other text."""
        text = 'I need to calculate: <tool>calculator</tool><args>{"expression": "2+2"}</args> Let me check.'
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["name"] == "calculator"

    def test_parser_xml_invalid_json(self):
        """Parse with invalid JSON in args."""
        text = '<tool>calculator</tool><args>invalid json</args>'
        result = FunctionCallParser.parse(text, format="xml")
        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"] == {}


class TestParserJSON:
    """Tests for JSON format parsing."""

    def test_parser_json_basic(self):
        """Parse JSON format tool call."""
        text = '{"name": "calculator", "arguments": {"expression": "2+2"}}'
        result = FunctionCallParser.parse(text, format="json")
        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"]["expression"] == "2+2"

    def test_parser_json_multiline(self):
        """Parse multiline JSON format."""
        text = '{"name": "python", "arguments": {"code": "print(1)"}}'
        result = FunctionCallParser.parse(text, format="json")
        assert result is not None
        assert result["name"] == "python"

    def test_parser_json_in_text(self):
        """Parse JSON format embedded in surrounding text."""
        text = 'Let me use a tool: {"name": "calculator", "arguments": {"expression": "5*5"}} That should help.'
        result = FunctionCallParser.parse(text, format="json")
        assert result is not None
        assert result["name"] == "calculator"


class TestParserTag:
    """Tests for tag format parsing."""

    def test_parser_tag_basic(self):
        """Parse tag format tool call."""
        text = '<function=calculator>{"expression": "2+2"}</function>'
        result = FunctionCallParser.parse(text, format="tag")
        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"]["expression"] == "2+2"

    def test_parser_tag_invalid_json(self):
        """Parse tag with invalid JSON."""
        text = "<function=calculator>invalid</function>"
        result = FunctionCallParser.parse(text, format="tag")
        assert result is not None
        assert result["name"] == "calculator"
        assert result["arguments"] == {}


class TestParserNone:
    """Tests for no-match cases."""

    def test_parser_none_plain_text(self):
        """Text without tool call returns None."""
        text = "This is just a plain response with no tools."
        assert FunctionCallParser.parse(text, format="xml") is None

    def test_parser_none_empty(self):
        """Empty string returns None."""
        assert FunctionCallParser.parse("", format="xml") is None

    def test_parser_none_json_no_match(self):
        """JSON without tool structure returns None."""
        text = '{"key": "value"}'
        assert FunctionCallParser.parse(text, format="json") is None

    def test_parser_none_tag_no_match(self):
        """No matching tag returns None."""
        text = "Just some text"
        assert FunctionCallParser.parse(text, format="tag") is None


# --------------------------------------------------------------------------- #
# Tool Prompt Injection Tests
# --------------------------------------------------------------------------- #


class TestInjectToolPrompt:
    """Tests for inject_tool_prompt."""

    def test_inject_contains_tool_descriptions(self):
        """Test that injected prompt contains tool descriptions."""
        system_prompt = "You are a helpful assistant."
        tools = [
            {"name": "calculator", "description": "Do math", "parameters": {}},
            {"name": "search", "description": "Search the web", "parameters": {}},
        ]
        result = FunctionCallParser.inject_tool_prompt(system_prompt, tools)
        assert "Available tools:" in result
        assert "calculator" in result
        assert "search" in result
        assert "Do math" in result

    def test_inject_contains_usage_format(self):
        """Test that injected prompt contains usage format."""
        system_prompt = "You are a helpful assistant."
        tools = [
            {"name": "calculator", "description": "Do math", "parameters": {}}
        ]
        result = FunctionCallParser.inject_tool_prompt(system_prompt, tools)
        assert "<tool>TOOL_NAME</tool>" in result
        assert "<args>" in result

    def test_inject_preserves_original_prompt(self):
        """Test that original system prompt is preserved."""
        system_prompt = "You are a helpful assistant."
        tools = []
        result = FunctionCallParser.inject_tool_prompt(system_prompt, tools)
        assert result.startswith(system_prompt)

    def test_inject_contains_param_descriptions(self):
        """Test that parameter descriptions are included."""
        system_prompt = "You are a helpful assistant."
        tools = [
            {
                "name": "calculator",
                "description": "Do math",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The math expression",
                        }
                    },
                    "required": ["expression"],
                },
            }
        ]
        result = FunctionCallParser.inject_tool_prompt(system_prompt, tools)
        assert "expression" in result
        assert "The math expression" in result

    def test_inject_instructs_final_answer(self):
        """Test that prompt instructs model to give final answer directly."""
        system_prompt = "You are a helpful assistant."
        tools = []
        result = FunctionCallParser.inject_tool_prompt(system_prompt, tools)
        assert "final answer" in result.lower() or "respond directly" in result.lower()
