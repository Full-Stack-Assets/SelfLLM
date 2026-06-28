"""Tests for the ReAct agent loop.

Tests cover:
- Agent initialization with correct defaults
- Agent run without tools (direct answer)
- Agent run with tool use (calculator)
- Agent max iterations limit
- Agent reasoning trace
- Streaming agent run
"""

from typing import List

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.tools.agent import Agent
from selfllm.tools.registry import ToolRegistry


# --------------------------------------------------------------------------- #
# Mock model helpers
# --------------------------------------------------------------------------- #


class MockModel:
    """Mock model that returns predetermined responses."""

    def __init__(self, responses: List[str] = None, tokenizer=None):
        self.responses = responses or []
        self.call_count = 0
        self.tokenizer = tokenizer

    def generate(self, prompt_ids, **kwargs):
        """Return predetermined response."""
        # Get the next response
        if self.call_count < len(self.responses):
            response_text = self.responses[self.call_count]
        else:
            response_text = "I don't know."
        self.call_count += 1

        # Encode response using the provided tokenizer, or use a simple fallback
        if self.tokenizer is not None:
            response_ids = self.tokenizer.encode(response_text)
        else:
            # Fallback: simple character encoding
            response_ids = [ord(c) % 256 for c in response_text]

        all_ids = prompt_ids[0].tolist() + response_ids

        result = {
            "sequences": torch.tensor([all_ids]),
            "scores": torch.tensor([[0.9]]),
        }
        return result


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self):
        self._vocab = {}
        self._next_id = 10
        # Pre-populate common tokens
        for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            self._vocab[char] = self._next_id
            self._next_id += 1
        for token in [" ", "\n", "<", ">", "/", "{", "}", '"', ":", ",", ".", "-"]:
            self._vocab[token] = self._next_id
            self._next_id += 1

    def encode(self, text: str) -> List[int]:
        ids = []
        for char in text:
            if char not in self._vocab:
                self._vocab[char] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[char])
        return ids

    def decode(self, token_ids: List[int]) -> str:
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv.get(i, "") for i in token_ids)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tokenizer():
    """Return a mock tokenizer."""
    return MockTokenizer()


@pytest.fixture
def tool_registry():
    """Return a tool registry."""
    return ToolRegistry()


# --------------------------------------------------------------------------- #
# Agent Init Tests
# --------------------------------------------------------------------------- #


class TestAgentInit:
    """Tests for agent initialization."""

    def test_agent_init_defaults(self, tokenizer, tool_registry):
        """Test agent initialization with correct defaults."""
        mock_model = MockModel()
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
        )
        assert agent.model is mock_model
        assert agent.tokenizer is tokenizer
        assert agent.tools is tool_registry
        assert agent.max_iterations == 10
        assert agent.max_tokens_per_step == 256
        assert agent.temperature == 0.7
        assert agent.executor is not None

    def test_agent_init_custom_params(self, tokenizer):
        """Test agent with custom parameters."""
        mock_model = MockModel()
        registry = ToolRegistry()
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=registry,
            max_iterations=5,
            max_tokens_per_step=128,
            temperature=0.5,
            system_prompt="Custom prompt",
        )
        assert agent.max_iterations == 5
        assert agent.max_tokens_per_step == 128
        assert agent.temperature == 0.5
        assert agent.system_prompt == "Custom prompt"

    def test_agent_init_default_tools(self, tokenizer):
        """Test that default tools are created when none provided."""
        mock_model = MockModel()
        agent = Agent(model=mock_model, tokenizer=tokenizer)
        assert agent.tools is not None
        assert "calculator" in agent.tools


# --------------------------------------------------------------------------- #
# Agent Run Tests
# --------------------------------------------------------------------------- #


class TestAgentRun:
    """Tests for the agent run loop."""

    def test_agent_run_no_tools(self, tokenizer, tool_registry):
        """Test direct answer without tool use."""
        mock_model = MockModel(
            responses=["The answer is 42."],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("What is the answer?")
        assert result["finished"] is True
        assert result["iterations"] == 1
        assert "42" in result["answer"]
        assert len(result["tool_calls"]) == 0
        assert len(result["reasoning_trace"]) == 1

    def test_agent_run_with_tool(self, tokenizer, tool_registry):
        """Test agent that uses calculator tool then answers."""
        # First response: calls calculator, Second: provides answer
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "2 + 2"}</args>',
                "The answer is 4.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("What is 2 + 2?")
        assert result["finished"] is True
        assert result["iterations"] == 2
        assert len(result["tool_calls"]) == 1
        # Check the tool call was parsed correctly
        assert result["tool_calls"][0]["call"]["name"] == "calculator"
        # Check tool result
        assert result["tool_calls"][0]["result"]["status"] == "success"
        assert "4" in result["tool_calls"][0]["result"]["result"]

    def test_agent_run_multiple_tools(self, tokenizer, tool_registry):
        """Test agent that uses multiple tools in sequence."""
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "1 + 1"}</args>',
                '<tool>count_words</tool><args>{"text": "hello world"}</args>',
                "Done with calculations.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("Do some work")
        assert result["finished"] is True
        assert result["iterations"] == 3
        assert len(result["tool_calls"]) == 2

    def test_agent_run_tool_error(self, tokenizer, tool_registry):
        """Test agent handles tool errors gracefully."""
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "1/0"}</args>',
                "Let me try something else.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("Calculate something")
        assert result["finished"] is True
        # The calculator tool catches ZeroDivisionError internally
        assert result["tool_calls"][0]["result"]["status"] == "success"
        # But the result text contains Error:
        assert "Error" in result["tool_calls"][0]["result"]["result"]

    def test_agent_max_iterations(self, tokenizer, tool_registry):
        """Test agent stops at max iterations limit."""
        # Model always wants to use tools, never answers
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "1+1"}</args>'
                for _ in range(15)
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            max_iterations=3,
            device="cpu",
        )
        result = agent.run("Calculate something")
        assert result["finished"] is False
        assert result["iterations"] == 3
        assert len(result["tool_calls"]) == 3
        assert "maximum" in result["answer"].lower()

    def test_agent_reasoning_trace(self, tokenizer, tool_registry):
        """Test that reasoning trace is populated."""
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "5 * 5"}</args>',
                "The result is 25.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("What is 5 * 5?")
        assert len(result["reasoning_trace"]) == 2
        assert "Step 1" in result["reasoning_trace"][0]
        assert "Step 2" in result["reasoning_trace"][1]


# --------------------------------------------------------------------------- #
# Agent Streaming Tests
# --------------------------------------------------------------------------- #


class TestAgentStream:
    """Tests for the streaming agent."""

    def test_agent_stream_no_tools(self, tokenizer, tool_registry):
        """Test streaming with direct answer."""
        mock_model = MockModel(responses=["Hello there!"], tokenizer=tokenizer)
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        events = list(agent.run_stream("Say hi"))
        assert len(events) == 1
        assert events[0]["type"] == "final"
        assert events[0]["finished"] is True

    def test_agent_stream_with_tool(self, tokenizer, tool_registry):
        """Test streaming with tool use."""
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "2+2"}</args>',
                "It's 4.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        events = list(agent.run_stream("What is 2+2?"))
        assert len(events) == 2
        assert events[0]["type"] == "tool_call"
        assert events[0]["call"]["name"] == "calculator"
        assert events[1]["type"] == "final"
        assert events[1]["finished"] is True


# --------------------------------------------------------------------------- #
# Integration with real model
# --------------------------------------------------------------------------- #


class TestAgentIntegration:
    """Integration tests using actual model (tiny config)."""

    @pytest.fixture
    def tiny_config(self):
        """Return a tiny model config."""
        return ModelConfig(
            vocab_size=256,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=128,
            max_seq_len=64,
            dropout=0.0,
            tie_weights=True,
            use_rope=True,
            use_swiglu=True,
        )

    @pytest.fixture
    def real_model(self, tiny_config):
        """Return an instantiated tiny model."""
        return SelfImprovingLLM(tiny_config)

    @pytest.fixture
    def real_tokenizer(self):
        """Return a trained tokenizer."""
        tok = BPETokenizer(vocab_size=256)
        tok.train(
            [
                "hello world",
                "calculator args tool",
                "The answer is",
                '<tool>calculator</tool><args>{"expression": "2+2"}</args>',
                "result",
            ]
        )
        return tok

    def test_agent_run_real_model_direct_answer(self, real_model, real_tokenizer):
        """Test agent with real model producing direct answer."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        real_model = real_model.to(device)
        agent = Agent(
            model=real_model,
            tokenizer=real_tokenizer,
            max_tokens_per_step=20,
            temperature=1.0,
            device=device,
        )
        # This won't have tool calls because the model is untrained
        result = agent.run("Hello")
        assert "iterations" in result
        assert result["iterations"] >= 1
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_agent_context_building(self, tokenizer, tool_registry):
        """Test that context is properly built across iterations."""
        mock_model = MockModel(
            responses=[
                '<tool>calculator</tool><args>{"expression": "1+1"}</args>',
                "The answer is 2.",
            ],
            tokenizer=tokenizer,
        )
        agent = Agent(
            model=mock_model,
            tokenizer=tokenizer,
            tools=tool_registry,
            device="cpu",
        )
        result = agent.run("What is 1+1?")

        # The model should have been called twice
        assert mock_model.call_count == 2

        # Should have tool result in context for second call
        assert result["finished"] is True
        assert len(result["tool_calls"]) == 1
