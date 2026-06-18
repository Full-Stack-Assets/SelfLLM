"""Quality filtering and scoring for self-generated training data."""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.quality_model import QualityModel
from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class QualityFilter:
    """Multi-dimensional quality filter for self-generated training data.

    Filters on: perplexity, diversity, coherence, length appropriateness.
    Uses a composite scoring formula to rank and filter samples.
    """

    def __init__(
        self,
        quality_model: Optional[QualityModel] = None,
        base_model: Optional[SelfImprovingLLM] = None,
        tokenizer: Optional[BPETokenizer] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the quality filter.

        Args:
            quality_model: QualityModel for neural quality scoring.
            base_model: SelfImprovingLLM for perplexity computation.
            tokenizer: BPETokenizer for encoding text.
            device: Device to run computations on ('cuda' or 'cpu').
        """
        self.quality_model = quality_model
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.device = device

    def _get_max_seq_len(self) -> int:
        """Get max sequence length from base model config."""
        if self.base_model is not None:
            config = getattr(self.base_model, "config", None)
            if config is not None:
                return getattr(config, "max_seq_len", 512)
        return 512

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text under the base model.

        Uses a forward pass to compute cross-entropy loss and converts
        to perplexity. Lower is better.

        Args:
            text: Input text to score.

        Returns:
            Perplexity value (float). Returns a large value on error.
        """
        if self.base_model is None or self.tokenizer is None:
            return 100.0

        try:
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 2:
                return 100.0

            # Truncate to model's max sequence length to avoid size mismatch
            max_len = self._get_max_seq_len()
            token_ids = token_ids[:max_len]

            ids_tensor = torch.tensor([token_ids], device=self.device)

            with torch.no_grad():
                outputs = self.base_model(ids_tensor)
                logits = outputs["logits"]

                # Shift logits and targets for next-token prediction
                # logits: [1, seq_len, vocab_size], targets: [1, seq_len]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_targets = ids_tensor[:, 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_targets.view(-1),
                    reduction="mean",
                )
                perplexity = math.exp(loss.item())

            return min(perplexity, 1000.0)  # Cap to avoid overflow

        except Exception as e:
            logger.warning(f"Perplexity computation failed: {e}")
            return 100.0

    def compute_diversity_score(self, text: str) -> float:
        """Compute lexical diversity score.

        Uses unique trigram ratio combined with entropy of the token
        (word) distribution. Returns score in [0, 1].

        Args:
            text: Input text to score.

        Returns:
            Diversity score in [0, 1]. Higher is more diverse.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        words = text.lower().split()
        if len(words) < 3:
            return 0.0

        # Unique trigram ratio
        trigrams = []
        for i in range(len(words) - 2):
            trigram = tuple(words[i : i + 3])
            trigrams.append(trigram)

        if not trigrams:
            return 0.0

        unique_trigrams = len(set(trigrams))
        total_trigrams = len(trigrams)
        trigram_ratio = unique_trigrams / total_trigrams if total_trigrams > 0 else 0.0

        # Entropy of token distribution
        from collections import Counter

        word_counts = Counter(words)
        total_words = len(words)
        entropy = 0.0
        for count in word_counts.values():
            p = count / total_words
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize entropy: max entropy for uniform distribution is log2(vocab_size)
        max_entropy = math.log2(max(len(word_counts), 2))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        # Combine: 60% trigram ratio, 40% entropy
        diversity = 0.6 * trigram_ratio + 0.4 * normalized_entropy
        return float(max(0.0, min(1.0, diversity)))

    def compute_coherence_score(self, text: str) -> float:
        """Estimate coherence via sentence embedding similarity.

        Uses mean pooling of word (token) embeddings as a proxy for
        sentence embeddings. Computes cosine similarity between consecutive
        sentence embeddings to measure topic coherence.

        Args:
            text: Input text to score.

        Returns:
            Coherence score in [0, 1]. Higher is more coherent.
        """
        if not text or len(text.strip()) == 0:
            return 0.0

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 2:
            return 0.5  # Single sentence: neutral coherence

        # Simple embedding proxy: character-level bag-of-words vectors
        def _sentence_vector(sentence: str) -> torch.Tensor:
            """Create a simple sentence vector from character n-gram frequencies."""
            words = sentence.lower().split()
            if not words:
                return torch.zeros(128)

            # Build a fixed-size vector from word hash distributions
            vec = torch.zeros(128)
            for word in words:
                for i, ch in enumerate(word[:16]):
                    idx = (ord(ch) + i * 256) % 128
                    vec[idx] += 1.0

            # Normalize
            norm = vec.norm()
            if norm > 0:
                vec = vec / norm
            return vec

        sentence_vecs = [_sentence_vector(s) for s in sentences]

        # Compute cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(sentence_vecs) - 1):
            sim = F.cosine_similarity(
                sentence_vecs[i].unsqueeze(0),
                sentence_vecs[i + 1].unsqueeze(0),
                dim=1,
            )
            similarities.append(sim.item())

        if not similarities:
            return 0.5

        # Average similarity; scale to [0, 1] (cosine similarity is in [-1, 1])
        avg_sim = sum(similarities) / len(similarities)
        coherence = (avg_sim + 1.0) / 2.0  # Normalize to [0, 1]
        return float(max(0.0, min(1.0, coherence)))

    def compute_length_score(
        self,
        text: str,
        target_range: Tuple[int, int] = (50, 500),
    ) -> float:
        """Score text length appropriateness using a Gaussian around target midpoint.

        Args:
            text: Input text to score.
            target_range: Tuple of (min, max) target word count.

        Returns:
            Length score in [0, 1]. Peaks at midpoint of target range.
        """
        if not text:
            return 0.0

        word_count = len(text.split())
        min_len, max_len = target_range
        mid = (min_len + max_len) / 2.0

        # Gaussian: score = exp(-((x - mu)^2) / (2 * sigma^2))
        # sigma chosen so that at the boundaries score ~ 0.6
        sigma = (max_len - min_len) / 2.0
        if sigma <= 0:
            sigma = 1.0

        score = math.exp(-((word_count - mid) ** 2) / (2.0 * sigma**2))
        return float(max(0.0, min(1.0, score)))

    def filter_batch(
        self,
        samples: List[Dict[str, Any]],
        min_quality_score: float = 0.6,
        max_perplexity: float = 100.0,
        min_diversity: float = 0.3,
        keep_ratio: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Filter a batch of generated samples, keeping only high-quality ones.

        Scoring formula::
            overall_score = 0.3 * normalized_quality_model
                          + 0.3 * diversity
                          + 0.2 * coherence
                          + 0.2 * length_score

        Args:
            samples: List of sample dicts with 'response' text.
            min_quality_score: Minimum overall quality score to keep.
            max_perplexity: Maximum perplexity allowed.
            min_diversity: Minimum diversity score allowed.
            keep_ratio: Fraction of samples to keep (sorted by quality).

        Returns:
            Filtered and sorted list of samples with scores attached.
        """
        if not samples:
            return []

        logger.info(f"Filtering {len(samples)} samples...")

        scored_samples = []
        for sample in samples:
            response = sample.get("response", "")

            # Compute individual scores
            perplexity = self.compute_perplexity(response)
            diversity = self.compute_diversity_score(response)
            coherence = self.compute_coherence_score(response)
            length_score = self.compute_length_score(response)

            # Get quality model score if available
            quality_model_score = 0.5  # Default neutral score
            if self.quality_model is not None and self.tokenizer is not None:
                try:
                    token_ids = self.tokenizer.encode(response)
                    token_ids = token_ids[:512]  # Truncate
                    ids_tensor = torch.tensor([token_ids], device=self.device)
                    with torch.no_grad():
                        q_out = self.quality_model(ids_tensor)
                        quality_model_score = q_out["overall_score"][0].item()
                        # Normalize to [0, 1] if needed
                        quality_model_score = max(0.0, min(1.0, quality_model_score))
                except Exception as e:
                    logger.warning(f"Quality model scoring failed: {e}")

            # Normalize perplexity: lower is better, so invert and normalize
            # perplexity of 1 -> score 1.0, perplexity of 100 -> score 0.0
            normalized_perplexity = max(
                0.0, min(1.0, 1.0 - (math.log(perplexity) / math.log(1000.0)))
            ) if perplexity > 0 else 0.0

            # Composite scoring
            overall_score = (
                0.3 * quality_model_score
                + 0.3 * diversity
                + 0.2 * coherence
                + 0.2 * length_score
            )

            # Attach scores to sample
            scored_sample = dict(sample)
            scored_sample["quality_scores"] = {
                "overall": float(overall_score),
                "quality_model": float(quality_model_score),
                "diversity": float(diversity),
                "coherence": float(coherence),
                "length_score": float(length_score),
                "perplexity": float(perplexity),
                "normalized_perplexity": float(normalized_perplexity),
            }

            scored_samples.append(scored_sample)

        # Apply hard filters
        filtered = [
            s for s in scored_samples
            if s["quality_scores"]["perplexity"] <= max_perplexity
            and s["quality_scores"]["diversity"] >= min_diversity
            and s["quality_scores"]["overall"] >= min_quality_score
        ]

        logger.info(f"After hard filters: {len(filtered)} / {len(samples)}")

        # Sort by overall quality score (descending)
        filtered.sort(key=lambda x: x["quality_scores"]["overall"], reverse=True)

        # Keep top keep_ratio
        keep_count = max(1, int(len(samples) * keep_ratio))
        result = filtered[:keep_count]

        logger.info(f"Final kept: {len(result)} / {len(samples)}")
        if result:
            avg_score = sum(s["quality_scores"]["overall"] for s in result) / len(result)
            logger.info(f"Average quality score: {avg_score:.4f}")

        return result
