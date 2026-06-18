"""Constitutional AI: Self-critique and revision for self-improving LLMs.

Implements the Constitutional AI pipeline from Bai et al. (2022):
1. Generate an initial response to a prompt
2. Critique the response against constitutional principles
3. Generate a revised response incorporating the critique
4. Use the (initial, revised) pairs as preference data for training

This enables self-improvement without any human labels — the model
 critiques and improves its own outputs.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import torch

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class ConstitutionalAI:
    """Constitutional AI: Self-critique and revision.

    The model generates a response, critiques it against a set of
    constitutional principles, and then revises the response based on
    the critique. The revised response is preferred over the initial.

    Constitution principles cover:
    - Helpfulness, honesty, and harmlessness
    - Bias and stereotype detection
    - Factual accuracy
    - Potential for misuse or harm
    - Confidence calibration

    Args:
        model: The language model to use for generation.
        tokenizer: Tokenizer for encoding/decoding text.
        device: Device to run generation on.
        max_new_tokens: Maximum tokens to generate for responses.
        temperature: Sampling temperature for generation.
    """

    CONSTITUTION = [
        "Please critique the following response. Is it helpful, honest, and harmless? "
        "Identify any issues with clarity, accuracy, safety, or helpfulness.",
        "Does this response contain any biases, stereotypes, or harmful assumptions? "
        "Check for demographic biases, stereotyping, or unfair generalizations.",
        "Is this response factually accurate to the best of your knowledge? "
        "Point out any factual errors, unsupported claims, or misleading statements.",
        "Could this response be misinterpreted or used to cause harm? "
        "Consider potential misuse, dangerous applications, or harmful interpretations.",
        "Is the response appropriately calibrated in its confidence level? "
        "Should uncertain claims be qualified? Is the tone appropriate?",
    ]

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Ensure model is in eval mode for generation
        self.model.eval()

        logger.info(
            f"ConstitutionalAI initialized: device={device}, "
            f"max_new_tokens={max_new_tokens}, temperature={temperature}"
        )
        logger.info(f"Constitution has {len(self.CONSTITUTION)} principles")

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    def _generate_text(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate text from a prompt using the model.

        Args:
            prompt: The text prompt to generate from.
            max_new_tokens: Override for max tokens to generate.

        Returns:
            Generated text string (decoded, without the prompt).
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        # Generate
        with torch.no_grad():
            result = self.model.generate(
                prompt_tensor,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                top_k=50,
                top_p=0.95,
            )

        # Decode the full sequence and extract just the generated portion
        full_sequence = result["sequences"][0].tolist()
        generated_ids = full_sequence[len(prompt_ids):]

        if not generated_ids:
            return ""

        return self.tokenizer.decode(generated_ids)

    # ------------------------------------------------------------------ #
    # Core Constitutional AI pipeline
    # ------------------------------------------------------------------ #

    def generate_with_critique(self, prompt: str) -> Dict[str, str]:
        """Generate response, critique it, and revise.

        Process:
            1. Generate an initial response to the prompt.
            2. Select a random constitutional principle and generate a critique.
            3. Generate a revised response that addresses the critique.

        Args:
            prompt: The user's prompt/query.

        Returns:
            Dictionary with keys:
                - ``initial``: The initial model response.
                - ``critique``: The critique of the initial response.
                - ``revised``: The revised response incorporating the critique.
        """
        # Step 1: Generate initial response
        initial_response = self._generate_text(prompt)

        # Step 2: Select a random constitutional principle and critique
        critique_principle = random.choice(self.CONSTITUTION)
        critique_prompt = (
            f"User prompt: {prompt}\n\n"
            f"Assistant response: {initial_response}\n\n"
            f"Critique instruction: {critique_principle}\n\n"
            f"Critique:"
        )
        critique = self._generate_text(critique_prompt, max_new_tokens=self.max_new_tokens)

        # Step 3: Generate revised response incorporating critique
        revision_prompt = (
            f"User prompt: {prompt}\n\n"
            f"Original response: {initial_response}\n\n"
            f"Critique: {critique}\n\n"
            f"Based on the critique above, provide an improved response."
            f"Address all the issues raised while maintaining helpfulness and accuracy.\n\n"
            f"Improved response:"
        )
        revised_response = self._generate_text(
            revision_prompt, max_new_tokens=self.max_new_tokens
        )

        return {
            "initial": initial_response,
            "critique": critique,
            "revised": revised_response,
        }

    # ------------------------------------------------------------------ #
    # Training pair generation
    # ------------------------------------------------------------------ #

    def generate_training_pairs(
        self,
        prompts: List[str],
        num_pairs: int,
    ) -> List[Dict[str, str]]:
        """Generate (initial, revised) preference pairs for training.

        For each prompt, runs the Constitutional AI pipeline and produces
        a preference pair where the revised response is preferred over
        the initial response.

        Args:
            prompts: List of prompt strings.
            num_pairs: Number of preference pairs to generate. Will sample
                from prompts with replacement if num_pairs > len(prompts).

        Returns:
            List of dicts, each with keys:
                - ``prompt``: The original prompt.
                - ``chosen``: The revised (preferred) response.
                - ``rejected``: The initial (less preferred) response.
        """
        if not prompts:
            logger.warning("Empty prompts list provided")
            return []

        pairs: List[Dict[str, str]] = []

        # Sample prompts (with replacement if needed)
        sampled_prompts = random.choices(prompts, k=num_pairs)

        for i, prompt in enumerate(sampled_prompts):
            logger.info(f"Generating training pair {i + 1}/{num_pairs}")

            try:
                result = self.generate_with_critique(prompt)

                # Create preference pair: revised is chosen, initial is rejected
                pair = {
                    "prompt": prompt,
                    "chosen": result["revised"],
                    "rejected": result["initial"],
                }
                pairs.append(pair)

                logger.debug(
                    f"Pair {i + 1}: initial_len={len(result['initial'])}, "
                    f"revised_len={len(result['revised'])}"
                )

            except Exception as e:
                logger.warning(f"Failed to generate pair {i + 1}: {e}")
                continue

        logger.info(f"Generated {len(pairs)}/{num_pairs} training pairs")
        return pairs

    def generate_self_critique_batch(
        self,
        prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """Generate full critique data (prompt, initial, critique, revised).

        This is useful for inspecting the Constitutional AI pipeline
        or for training on critique generation itself.

        Args:
            prompts: List of prompt strings.

        Returns:
            List of dicts with keys ``prompt``, ``initial``, ``critique``, ``revised``.
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate_with_critique(prompt)
                result["prompt"] = prompt
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to process prompt: {e}")
                continue
        return results
