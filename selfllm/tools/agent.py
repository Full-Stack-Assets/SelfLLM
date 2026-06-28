"""ReAct-style agent: Reasoning + Acting loop."""

import logging
from typing import Any, Dict, Optional

import torch

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.tools.parser import FunctionCallParser
from selfllm.tools.registry import ToolExecutor, ToolRegistry

logger = logging.getLogger(__name__)


class Agent:
    """
    ReAct agent: Reason + Act loop with tool use.

    The model iteratively:
    1. THINKS about what to do (reasoning)
    2. DECIDES to use a tool or answer (action)
    3. OBSERVES tool results
    4. REPEATS until answer or max iterations

    Inspired by ReAct (Yao et al., 2022) and Toolformer.
    """

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        tools: Optional[ToolRegistry] = None,
        max_iterations: int = 10,
        max_tokens_per_step: int = 256,
        temperature: float = 0.7,
        device: str = "cuda",
        system_prompt: str = "You are a helpful AI assistant with access to tools.",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools or ToolRegistry()
        self.executor = ToolExecutor(self.tools)
        self.max_iterations = max_iterations
        self.max_tokens_per_step = max_tokens_per_step
        self.temperature = temperature
        self.device = device
        self.system_prompt = system_prompt

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the ReAct agent loop on a query.

        Args:
            query: User query

        Returns:
            {
                "answer": str,              # Final answer text
                "tool_calls": List[Dict],   # All tool calls made
                "reasoning_trace": List[str], # Full reasoning log
                "iterations": int,          # Number of iterations used
                "finished": bool,           # Whether agent completed normally
            }
        """
        tool_calls = []
        reasoning_trace = []
        interactions = []

        # Initial system prompt with tool definitions
        system = FunctionCallParser.inject_tool_prompt(
            self.system_prompt, self.tools.list_tools()
        )

        # Build initial context
        context = f"{system}\n\nUser: {query}\nAssistant:"

        for iteration in range(self.max_iterations):
            # Generate response
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(context)], device=self.device
            )

            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_tokens_per_step,
                    temperature=self.temperature,
                    top_p=0.9,
                )

            # Decode the full sequence, then extract just the new response
            full_text = self.tokenizer.decode(output["sequences"][0].tolist())
            response = full_text[len(context) :].strip()

            reasoning_trace.append(f"Step {iteration + 1}: {response[:200]}")

            # Check if response contains a tool call
            tool_call = FunctionCallParser.parse(response, format="xml")

            if tool_call is None:
                # No tool call -- model provided direct answer
                return {
                    "answer": response,
                    "tool_calls": tool_calls,
                    "reasoning_trace": reasoning_trace,
                    "iterations": iteration + 1,
                    "finished": True,
                }

            # Execute tool
            result = self.executor.execute(tool_call)
            tool_calls.append(
                {
                    "call": tool_call,
                    "result": result,
                    "iteration": iteration + 1,
                }
            )

            # Append to context for next iteration
            interactions.append(f"\n{response}")
            interactions.append(f"\n[Tool result]: {result['result']}")

            context = (
                f"{system}\n\nUser: {query}\n" + "".join(interactions) + "\nAssistant:"
            )

        # Max iterations reached
        return {
            "answer": "I reached the maximum number of iterations. "
            "Here's what I found:\n" + "\n".join(reasoning_trace),
            "tool_calls": tool_calls,
            "reasoning_trace": reasoning_trace,
            "iterations": self.max_iterations,
            "finished": False,
        }

    def run_stream(self, query: str):
        """Streaming version of run(). Yields reasoning steps as they happen."""
        tool_calls = []
        reasoning_trace = []
        interactions = []

        system = FunctionCallParser.inject_tool_prompt(
            self.system_prompt, self.tools.list_tools()
        )

        context = f"{system}\n\nUser: {query}\nAssistant:"

        for iteration in range(self.max_iterations):
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(context)], device=self.device
            )

            with torch.no_grad():
                output = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_tokens_per_step,
                    temperature=self.temperature,
                    top_p=0.9,
                )

            full_text = self.tokenizer.decode(output["sequences"][0].tolist())
            response = full_text[len(context) :].strip()

            reasoning_trace.append(f"Step {iteration + 1}: {response[:200]}")

            # Check for tool call
            tool_call = FunctionCallParser.parse(response, format="xml")

            if tool_call is None:
                yield {
                    "type": "final",
                    "answer": response,
                    "tool_calls": tool_calls,
                    "reasoning_trace": reasoning_trace,
                    "iterations": iteration + 1,
                    "finished": True,
                }
                return

            # Execute tool
            result = self.executor.execute(tool_call)
            tool_calls.append(
                {
                    "call": tool_call,
                    "result": result,
                    "iteration": iteration + 1,
                }
            )

            yield {
                "type": "tool_call",
                "call": tool_call,
                "result": result,
                "iteration": iteration + 1,
            }

            interactions.append(f"\n{response}")
            interactions.append(f"\n[Tool result]: {result['result']}")

            context = (
                f"{system}\n\nUser: {query}\n" + "".join(interactions) + "\nAssistant:"
            )

        yield {
            "type": "final",
            "answer": "I reached the maximum number of iterations. "
            "Here's what I found:\n" + "\n".join(reasoning_trace),
            "tool_calls": tool_calls,
            "reasoning_trace": reasoning_trace,
            "iterations": self.max_iterations,
            "finished": False,
        }
