"""Chain-of-Thought reasoning generation and self-consistency voting."""

import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import torch

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer


class ChainOfThoughtPrompts:
    """CoT prompt templates for eliciting step-by-step reasoning."""

    TEMPLATES: List[str] = [
        "Let's think step by step. {problem}",
        "I'll work through this carefully. {problem}",
        "First, let me identify what's being asked. {problem}",
        "Let me break this down into steps. {problem}",
        "To solve this, I'll use a systematic approach. {problem}",
        "I'll analyze this problem methodically. {problem}",
        "Let me work through this reasoning carefully. {problem}",
        "Here's my approach to solving this. {problem}",
        "I'll tackle this step by step with clear reasoning. {problem}",
        "Let me think about this problem carefully before answering. {problem}",
        "First, I'll identify the key elements. Then: {problem}",
        "I'll reason through this systematically. {problem}",
    ]

    MATH_PROBLEMS: List[str] = [
        "What is 15 * 23?",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "Solve for x: 3x + 7 = 22",
        "What is the sum of the first 20 positive integers?",
        "A rectangle has length 8 and width 5. What is its area and perimeter?",
        "If 5 workers can build a wall in 10 days, how long for 10 workers?",
        "What is 25% of 480?",
        "Calculate the average of 12, 18, 24, 30, and 36.",
        "A car uses 8 liters per 100km. How much for 350km?",
        "If x/4 = 7, what is 2x + 3?",
    ]

    LOGIC_PROBLEMS: List[str] = [
        "If all roses are flowers and some flowers fade quickly, can we conclude some roses fade quickly?",
        "Three people need to cross a bridge. Their crossing times are 1, 2, and 5 minutes. What's the minimum time?",
        "If it rains, the ground gets wet. The ground is wet. Did it necessarily rain?",
        "All squares are rectangles. Are all rectangles squares? Explain.",
        "If A implies B and B implies C, does A imply C?",
        "A farmer has 17 sheep and all but 9 die. How many are left?",
    ]


class ChainOfThoughtGenerator:
    """Generate CoT reasoning traces and train on them."""

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_cot_response(
        self,
        problem: str,
        template_idx: int = 0,
        max_think_tokens: int = 256,
        max_answer_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Dict[str, str]:
        """
        Generate response in format:
        <think>step-by-step reasoning...</think><answer>final answer</answer>

        Args:
            problem: The problem text to solve.
            template_idx: Index of the prompt template to use.
            max_think_tokens: Maximum tokens for reasoning section.
            max_answer_tokens: Maximum tokens for answer section.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            Dictionary with "thinking", "answer", and "full_text" keys.
        """
        # Format prompt with template
        template = ChainOfThoughtPrompts.TEMPLATES[
            template_idx % len(ChainOfThoughtPrompts.TEMPLATES)
        ]
        prompt_text = template.format(problem=problem) + "\n<think>"

        prompt_ids = torch.tensor(
            [self.tokenizer.encode(prompt_text)], device=self.device
        )

        # Generate thinking + answer
        output = self.model.generate(
            prompt_ids,
            max_new_tokens=max_think_tokens + max_answer_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        full_text = self.tokenizer.decode(output["sequences"][0].tolist())
        generated = full_text[len(prompt_text) :]

        # Parse <think>...</think><answer>...</answer>
        think_match = re.search(r"(.*?)<\/think>", generated, re.DOTALL)
        answer_match = re.search(r"<answer>(.*?)<\/answer>", generated, re.DOTALL)

        thinking = think_match.group(1).strip() if think_match else generated[:500]
        answer = answer_match.group(1).strip() if answer_match else generated[-200:]

        return {"thinking": thinking, "answer": answer, "full_text": full_text}

    def self_consistency_vote(
        self,
        problem: str,
        num_paths: int = 5,
        temperature: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Generate N reasoning paths, extract final answers, pick majority by exact match.

        Uses normalized string comparison (lowercase, no whitespace) for matching.

        Args:
            problem: The problem text to solve.
            num_paths: Number of reasoning paths to generate.
            temperature: Sampling temperature for diversity.

        Returns:
            Dictionary with "answer" (most common), "confidence" (frequency ratio),
            "all_paths" (list of all reasoning traces), and "answer_counts".
        """
        paths = []
        for i in range(num_paths):
            response = self.generate_cot_response(
                problem, template_idx=i, temperature=temperature
            )
            paths.append(response)

        # Count answer frequencies (normalize whitespace for matching)
        answers = [
            p["answer"].strip().lower().replace(" ", "") for p in paths
        ]
        counts = Counter(answers)

        best_answer = counts.most_common(1)[0][0] if counts else ""
        confidence = counts.most_common(1)[0][1] / num_paths if counts else 0.0

        # Return the full text of the best answer
        best_idx = answers.index(best_answer) if best_answer in answers else 0

        return {
            "answer": paths[best_idx]["answer"],
            "confidence": confidence,
            "all_paths": paths,
            "answer_counts": dict(counts),
        }

    def generate_cot_training_data(
        self,
        num_samples: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Generate training examples with CoT reasoning.
        Uses math and logic problem templates.

        Args:
            num_samples: Number of training samples to generate.

        Returns:
            List of sample dicts with "prompt", "response", "cot_text", "answer",
            and "token_ids" keys.
        """
        problems = (
            ChainOfThoughtPrompts.MATH_PROBLEMS
            + ChainOfThoughtPrompts.LOGIC_PROBLEMS
        )

        samples = []
        for i in range(num_samples):
            problem = problems[i % len(problems)]
            result = self.generate_cot_response(problem, template_idx=i)

            cot_template = ChainOfThoughtPrompts.TEMPLATES[
                i % len(ChainOfThoughtPrompts.TEMPLATES)
            ]
            prompt = cot_template.format(problem=problem)
            full_response = (
                f"<think>{result['thinking']}</think>"
                f"<answer>{result['answer']}</answer>"
            )

            samples.append(
                {
                    "prompt": prompt,
                    "response": full_response,
                    "cot_text": result["thinking"],
                    "answer": result["answer"],
                    "token_ids": self.tokenizer.encode(prompt)
                    + self.tokenizer.encode(full_response),
                }
            )

        return samples
