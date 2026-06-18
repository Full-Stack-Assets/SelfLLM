"""Evaluation suite for tracking model quality across recursive self-improvement iterations."""

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class SelfImprovementEvaluator:
    """Evaluates model quality and tracks improvement across recursive iterations.

    Performs multiple evaluation dimensions:
    - Perplexity on evaluation texts (lower is better)
    - Generation quality (length, diversity, coherence, repetition)
    - Self-consistency (agreement across multiple generations)
    """

    def __init__(
        self,
        model: "SelfImprovingLLM",  # noqa: F821
        tokenizer: "BPETokenizer",  # noqa: F821
        eval_prompts: List[str],
        device: str = "cuda",
    ):
        """Initialize the evaluator.

        Args:
            model: The language model to evaluate.
            tokenizer: The tokenizer for encoding/decoding.
            eval_prompts: Fixed set of prompts for consistent evaluation.
            device: Device to run evaluation on ("cuda" or "cpu").
        """
        self.model = model
        self.tokenizer = tokenizer
        self.eval_prompts = eval_prompts
        self.device = device

        self.model.to(self.device)
        self.model.eval()

    def evaluate_perplexity(self, texts: List[str]) -> float:
        """Compute average perplexity on evaluation texts.

        Args:
            texts: List of text strings to evaluate perplexity on.

        Returns:
            Average perplexity score (lower is better).
        """
        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            for text in texts:
                token_ids = self.tokenizer.encode(text)
                if len(token_ids) < 2:
                    continue

                input_ids = torch.tensor([token_ids], device=self.device)
                outputs = self.model(input_ids)
                logits = outputs["logits"]

                # Shift logits and targets for next-token prediction
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = input_ids[:, 1:].contiguous()

                # Compute cross-entropy loss
                nll = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction="sum",
                )

                total_nll += nll.item()
                total_tokens += shift_targets.numel()

        if total_tokens == 0:
            return float("inf")

        avg_nll = total_nll / total_tokens
        perplexity = math.exp(avg_nll)
        return perplexity

    def evaluate_generation_quality(
        self, num_samples: int = 50
    ) -> Dict[str, float]:
        """Evaluate quality of generated outputs on fixed prompts.

        Generates responses to a subset of eval prompts and scores them
        on multiple quality dimensions.

        Args:
            num_samples: Number of prompts to evaluate (sampled from eval_prompts).

        Returns:
            Dict with keys: avg_length, diversity_score, coherence_score,
            repetition_rate, overall_quality.
        """
        prompts = self.eval_prompts[:num_samples]
        if not prompts:
            return {
                "avg_length": 0.0,
                "diversity_score": 0.0,
                "coherence_score": 0.0,
                "repetition_rate": 0.0,
                "overall_quality": 0.0,
            }

        self.model.eval()
        generated_texts: List[str] = []

        with torch.no_grad():
            for prompt in prompts:
                prompt_ids = torch.tensor(
                    [self.tokenizer.encode(prompt)], device=self.device
                )
                outputs = self.model.generate(
                    prompt_ids,
                    max_new_tokens=128,
                    temperature=0.8,
                    top_p=0.92,
                )
                sequences = outputs["sequences"][0].cpu().tolist()
                generated = self.tokenizer.decode(sequences)
                # Remove the prompt portion
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                generated_texts.append(generated)

        # Compute metrics
        avg_length = self._compute_avg_length(generated_texts)
        diversity_score = self._compute_diversity(generated_texts)
        coherence_score = self._compute_coherence(generated_texts)
        repetition_rate = self._compute_repetition(generated_texts)

        # Overall quality: weighted composite
        overall_quality = (
            0.25 * min(avg_length / 100.0, 1.0)  # Reward length up to ~100 tokens
            + 0.30 * diversity_score
            + 0.25 * coherence_score
            + 0.20 * (1.0 - repetition_rate)  # Lower repetition is better
        )

        return {
            "avg_length": avg_length,
            "diversity_score": diversity_score,
            "coherence_score": coherence_score,
            "repetition_rate": repetition_rate,
            "overall_quality": overall_quality,
        }

    def evaluate_self_consistency(self, num_problems: int = 20) -> float:
        """Measure self-consistency by asking the same question multiple times.

        Generates 5 responses per problem and measures pairwise similarity
        using n-gram overlap as a proxy for embedding similarity.

        Args:
            num_problems: Number of eval prompts to test consistency on.

        Returns:
            Self-consistency score in [0, 1] where 1 means perfect agreement.
        """
        prompts = self.eval_prompts[:num_problems]
        if not prompts:
            return 0.0

        self.model.eval()
        num_generations = 5
        consistency_scores = []

        with torch.no_grad():
            for prompt in prompts:
                responses: List[str] = []
                for _ in range(num_generations):
                    prompt_ids = torch.tensor(
                        [self.tokenizer.encode(prompt)], device=self.device
                    )
                    outputs = self.model.generate(
                        prompt_ids,
                        max_new_tokens=64,
                        temperature=0.8,
                        top_p=0.92,
                    )
                    sequences = outputs["sequences"][0].cpu().tolist()
                    generated = self.tokenizer.decode(sequences)
                    if generated.startswith(prompt):
                        generated = generated[len(prompt):].strip()
                    responses.append(generated)

                # Compute pairwise similarity between responses
                similarities = []
                for i in range(len(responses)):
                    for j in range(i + 1, len(responses)):
                        sim = self._text_similarity(responses[i], responses[j])
                        similarities.append(sim)

                if similarities:
                    avg_sim = sum(similarities) / len(similarities)
                    consistency_scores.append(avg_sim)

        if not consistency_scores:
            return 0.0

        return sum(consistency_scores) / len(consistency_scores)

    def compute_improvement_delta(
        self,
        current_metrics: Dict[str, float],
        previous_metrics: Optional[Dict[str, float]],
    ) -> float:
        """Compute composite improvement score between two evaluation runs.

        Positive values indicate improvement, negative values indicate degradation.

        Args:
            current_metrics: Metrics from the current iteration.
            previous_metrics: Metrics from the previous iteration (None for first).

        Returns:
            Composite improvement delta. Positive = improved.
        """
        if previous_metrics is None:
            return 0.0  # First iteration has no baseline

        # Perplexity change: lower is better, so negate
        current_ppl = current_metrics.get("perplexity", 0.0)
        previous_ppl = previous_metrics.get("perplexity", 1.0)
        if previous_ppl > 0 and current_ppl > 0:
            perplexity_change = (previous_ppl - current_ppl) / previous_ppl
        else:
            perplexity_change = 0.0

        # Quality change: higher is better
        current_quality = current_metrics.get("overall_quality", 0.0)
        previous_quality = previous_metrics.get("overall_quality", 0.0)
        if previous_quality > 0:
            quality_change = (current_quality - previous_quality) / previous_quality
        else:
            quality_change = current_quality

        # Consistency change: higher is better
        current_consistency = current_metrics.get("self_consistency", 0.0)
        previous_consistency = previous_metrics.get("self_consistency", 0.0)
        if previous_consistency > 0:
            consistency_change = (
                current_consistency - previous_consistency
            ) / previous_consistency
        else:
            consistency_change = current_consistency

        # Weighted composite: 0.4 * perplexity + 0.3 * quality + 0.3 * consistency
        # Normalize so positive = improvement
        improvement_delta = (
            0.4 * perplexity_change
            + 0.3 * quality_change
            + 0.3 * consistency_change
        )

        return improvement_delta

    def full_evaluation(self) -> Dict[str, float]:
        """Run complete evaluation suite.

        Returns:
            Combined metrics dict with perplexity, generation quality metrics,
            self-consistency, and timestamp.
        """
        # Evaluate perplexity on eval prompts
        perplexity = self.evaluate_perplexity(self.eval_prompts)

        # Evaluate generation quality
        quality_metrics = self.evaluate_generation_quality(
            num_samples=min(50, len(self.eval_prompts))
        )

        # Evaluate self-consistency
        consistency = self.evaluate_self_consistency(
            num_problems=min(20, len(self.eval_prompts))
        )

        metrics = {
            "perplexity": perplexity,
            "avg_length": quality_metrics["avg_length"],
            "diversity_score": quality_metrics["diversity_score"],
            "coherence_score": quality_metrics["coherence_score"],
            "repetition_rate": quality_metrics["repetition_rate"],
            "overall_quality": quality_metrics["overall_quality"],
            "self_consistency": consistency,
        }

        return metrics

    # --- Helper methods ---

    def _compute_avg_length(self, texts: List[str]) -> float:
        """Compute average token count of texts."""
        if not texts:
            return 0.0
        lengths = [len(text.split()) for text in texts]
        return sum(lengths) / len(lengths)

    def _compute_diversity(self, texts: List[str]) -> float:
        """Compute lexical diversity using unique n-gram ratio.

        Returns score in [0, 1] where higher is more diverse.
        """
        if not texts:
            return 0.0

        total_diversity = 0.0
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < 2:
                continue

            # Bigram diversity
            bigrams = list(zip(tokens[:-1], tokens[1:]))
            if bigrams:
                unique_bigrams = len(set(bigrams))
                total_diversity += unique_bigrams / len(bigrams)

        return total_diversity / len(texts) if texts else 0.0

    def _compute_coherence(self, texts: List[str]) -> float:
        """Estimate coherence via sentence-level similarity.

        Uses mean word overlap between consecutive sentences as a proxy
        for coherence.

        Returns score in [0, 1].
        """
        if not texts:
            return 0.0

        total_coherence = 0.0
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]

            if len(sentences) < 2:
                total_coherence += 1.0  # Single sentence is coherent by default
                continue

            similarities = []
            for i in range(len(sentences) - 1):
                sim = self._text_similarity(sentences[i], sentences[i + 1])
                similarities.append(sim)

            total_coherence += sum(similarities) / len(similarities)

        return total_coherence / len(texts) if texts else 0.0

    def _compute_repetition(self, texts: List[str]) -> float:
        """Compute repetition rate as fraction of repeated n-grams.

        Returns score in [0, 1] where higher means more repetition.
        """
        if not texts:
            return 0.0

        total_repetition = 0.0
        for text in texts:
            tokens = text.lower().split()
            if len(tokens) < 4:
                continue

            # Count trigram repetitions
            trigrams = list(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
            if not trigrams:
                continue

            trigram_counts = Counter(trigrams)
            repeated = sum(1 for count in trigram_counts.values() if count > 1)
            total_repetition += repeated / len(trigrams)

        return total_repetition / len(texts) if texts else 0.0

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts using token overlap.

        Uses a Jaccard-like similarity on sets of tokens as a lightweight
        proxy for embedding cosine similarity.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score in [0, 1].
        """
        tokens_a = set(text_a.lower().split())
        tokens_b = set(text_b.lower().split())

        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)
