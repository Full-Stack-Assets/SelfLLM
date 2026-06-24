"""Debate agent: an LLM wrapped with a distinct perspective and sampling params."""

import re
from typing import List

import torch

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer


class DebateAgent:
    """A single debater wrapping a (model, tokenizer) with a fixed perspective.

    Multiple agents may share one underlying ``model`` instance while differing
    only by their ``perspective`` prompt prefix and sampling temperature, which
    is far cheaper than instantiating one model per agent. Passing distinct
    models is also supported simply by constructing each agent with its own.

    Args:
        name: Human-readable identifier for the agent (used in transcripts).
        model: The language model used for generation.
        tokenizer: Tokenizer providing ``encode``/``decode``/``eos_token_id``.
        perspective: A system/role prompt prefix describing this agent's stance
            (e.g. ``"You are a skeptical analyst."``). Injected into every prompt.
        temperature: Sampling temperature for this agent's generations.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling cutoff.
        max_new_tokens: Maximum number of new tokens to generate per call.
        device: Device on which prompt tensors are placed.
    """

    def __init__(
        self,
        name: str,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        perspective: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_new_tokens: int = 48,
        device: str = "cpu",
    ) -> None:
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.perspective = perspective
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.device = device

    # ------------------------------------------------------------------ #
    # Prompt construction
    # ------------------------------------------------------------------ #

    def build_propose_prompt(self, question: str) -> str:
        """Build the prompt asking this agent to propose an initial answer.

        Args:
            question: The debate question.

        Returns:
            The fully formatted prompt string, including this agent's perspective.
        """
        return (
            f"{self.perspective}\n"
            f"Question: {question}\n"
            f"Answer:"
        )

    def build_critique_prompt(self, question: str, other_answer: str) -> str:
        """Build the prompt asking this agent to critique a peer's answer.

        Args:
            question: The debate question.
            other_answer: Another agent's current answer to critique.

        Returns:
            The fully formatted critique prompt, including this agent's perspective.
        """
        return (
            f"{self.perspective}\n"
            f"Question: {question}\n"
            f"A peer answered: {other_answer}\n"
            f"Critique:"
        )

    def build_revise_prompt(
        self, question: str, own_answer: str, critiques: List[str]
    ) -> str:
        """Build the prompt asking this agent to revise its own answer.

        Args:
            question: The debate question.
            own_answer: This agent's previous answer.
            critiques: Critiques of this agent's answer raised by peers.

        Returns:
            The fully formatted revision prompt, including this agent's perspective.
        """
        joined = " ".join(c.strip() for c in critiques if c.strip())
        return (
            f"{self.perspective}\n"
            f"Question: {question}\n"
            f"Your previous answer: {own_answer}\n"
            f"Critiques: {joined}\n"
            f"Revised answer:"
        )

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    def _generate(self, prompt: str) -> str:
        """Encode ``prompt``, run the model, and decode only the new tokens.

        Args:
            prompt: The prompt text to condition on.

        Returns:
            The decoded generated continuation (prompt stripped, whitespace trimmed).
        """
        prompt_ids = torch.tensor(
            [self.tokenizer.encode(prompt)], device=self.device
        )
        prompt_len = prompt_ids.shape[1]

        output = self.model.generate(
            prompt_ids,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            stop_token_id=self.tokenizer.eos_token_id,
        )

        new_token_ids = output["sequences"][0][prompt_len:].tolist()
        text = self.tokenizer.decode(new_token_ids)
        return self._clean(text)

    @staticmethod
    def _clean(text: str) -> str:
        """Normalise generated text: drop the EOS marker and trim whitespace."""
        text = text.replace(BPETokenizer.EOS_TOKEN, " ")
        return re.sub(r"\s+", " ", text).strip()

    # ------------------------------------------------------------------ #
    # Debate actions
    # ------------------------------------------------------------------ #

    def propose(self, question: str) -> str:
        """Generate an initial answer to ``question`` from this perspective.

        Args:
            question: The debate question.

        Returns:
            The proposed answer text.
        """
        return self._generate(self.build_propose_prompt(question))

    def critique(self, question: str, other_answer: str) -> str:
        """Generate a critique of a peer's answer.

        Args:
            question: The debate question.
            other_answer: The peer answer to critique.

        Returns:
            The critique text.
        """
        return self._generate(self.build_critique_prompt(question, other_answer))

    def revise(
        self, question: str, own_answer: str, critiques: List[str]
    ) -> str:
        """Generate a revised answer given peer critiques.

        Args:
            question: The debate question.
            own_answer: This agent's previous answer.
            critiques: Critiques raised against this agent's answer.

        Returns:
            The revised answer text.
        """
        return self._generate(
            self.build_revise_prompt(question, own_answer, critiques)
        )
