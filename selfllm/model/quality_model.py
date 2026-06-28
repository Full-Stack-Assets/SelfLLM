"""Quality scoring model built on top of SelfImprovingLLM hidden states."""

from typing import Dict, List

import torch
import torch.nn as nn

from .model import SelfImprovingLLM


class QualityModel(nn.Module):
    """Lightweight quality scoring model on top of LLM hidden states.

    Scores responses on multiple dimensions — coherence, diversity, and
    complexity — and returns an overall quality score in ``[0, 1]``.

    Architecture:
        LLM encoder (frozen) -> per-dimension linear heads -> sigmoid

    Args:
        base_model: The ``SelfImprovingLLM`` to extract hidden states from.
        num_dimensions: Number of scoring dimensions (default 3).
    """

    # Dimension names
    DIMENSION_NAMES: List[str] = ["coherence", "diversity", "complexity"]

    def __init__(
        self,
        base_model: SelfImprovingLLM,
        num_dimensions: int = 3,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.num_dimensions = num_dimensions

        # Freeze the base model by default
        for param in self.base_model.parameters():
            param.requires_grad = False

        d_model = base_model.config.d_model

        # Per-dimension scoring heads: hidden_state -> score
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model // 4),
                    nn.SiLU(),
                    nn.Dropout(base_model.config.dropout),
                    nn.Linear(d_model // 4, 1),
                )
                for _ in range(num_dimensions)
            ]
        )

        # Overall score gating: combines dimension scores with learned weights
        self.score_gate = nn.Sequential(
            nn.Linear(num_dimensions, num_dimensions),
            nn.Sigmoid(),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Kaiming initialisation for linear layers in scoring heads."""
        for head in self.heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        for m in self.score_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Score input sequences on multiple quality dimensions.

        Args:
            token_ids: Input token IDs ``[batch, seq_len]``.

        Returns:
            Dictionary with keys:
                - ``overall_score``: ``[batch]`` float tensor in ``[0, 1]``
                - ``dimension_scores``: ``[batch, num_dimensions]`` dict of:
                    - coherence: logical consistency
                    - diversity: lexical/topic diversity
                    - complexity: appropriate complexity level
        """
        # Extract hidden states from base model (no gradient needed)
        with torch.no_grad():
            out = self.base_model(token_ids)
        hidden_states = out["hidden_states"]  # [batch, seq_len, d_model]

        # Pool hidden states: mean pooling over sequence length
        pooled = hidden_states.mean(dim=1)  # [batch, d_model]

        # Compute per-dimension scores
        dim_scores = torch.cat(
            [head(pooled) for head in self.heads], dim=-1
        )  # [batch, num_dimensions]
        dim_scores = torch.sigmoid(dim_scores)

        # Compute overall score with learned gating
        gate_weights = self.score_gate(dim_scores)  # [batch, num_dimensions]
        overall_score = (gate_weights * dim_scores).sum(dim=-1)  # [batch]
        overall_score = torch.sigmoid(overall_score)

        # Build dimension score dictionary
        dimension_scores: Dict[str, torch.Tensor] = {}
        for i, name in enumerate(self.DIMENSION_NAMES[: self.num_dimensions]):
            dimension_scores[name] = dim_scores[:, i]

        return {
            "overall_score": overall_score,
            "dimension_scores": dimension_scores,
        }

    def score_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        """Convenience method to get overall scores for a batch of sequences.

        Args:
            sequences: Token ID tensor ``[batch, seq_len]``.

        Returns:
            Overall quality scores ``[batch]`` in ``[0, 1]``.
        """
        return self.forward(sequences)["overall_score"]
