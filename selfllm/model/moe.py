"""Mixture of Experts (MoE) layer with top-k routing and load balancing.

Switch Transformer-style sparse MoE. Instead of one dense FFN per layer,
use N smaller "expert" FFNs + a learned router. Each token activates only
top-k experts, enabling massive model capacity with sub-linear compute.

Architecture::

    Router:    Linear(d_model, num_experts) -> softmax -> top-k
    Experts:   num_experts copies of FFN(d_model -> d_ff -> d_model)
    MoELayer:  Router + Expert dispatch / compute / combine

References:
    - Switch Transformers (Fedus et al., 2022)
    - GLaM (Du et al., 2022)
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class Router(nn.Module):
    """Learned router that assigns tokens to experts.

    Computes routing scores via a linear projection, applies softmax, and
    selects the top-k experts per token.  Also produces a load-balancing
    auxiliary loss that encourages uniform expert utilisation.

    Args:
        d_model: Token embedding dimension.
        num_experts: Number of expert FFNs available.
        top_k: How many experts each token is routed to.
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            x: Flattened token tensor ``[batch * seq_len, d_model]``.

        Returns:
            - **expert_indices** ``[B*T, top_k]`` -- expert IDs for each token.
            - **expert_weights** ``[B*T, top_k]`` -- normalised gating weights
              (sum to 1 per token).
            - **aux_loss** scalar -- load-balancing loss.
        """
        # Routing logits
        router_logits = self.gate(x)  # [B*T, num_experts]

        # Softmax to get routing weights
        weights = F.softmax(router_logits, dim=-1)  # [B*T, num_experts]

        # Top-k experts per token
        expert_weights, expert_indices = torch.topk(
            weights, self.top_k, dim=-1
        )  # [B*T, top_k]

        # Normalise the top-k weights so they sum to 1 per token
        expert_weights = expert_weights / expert_weights.sum(
            dim=-1, keepdim=True
        )

        # Load balancing auxiliary loss
        aux_loss = self._compute_load_balancing_loss(router_logits, expert_indices)

        return expert_indices, expert_weights, aux_loss

    def _compute_load_balancing_loss(
        self, router_logits: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load-balancing auxiliary loss.

        Encourages uniform expert utilisation::

            f_i = fraction of (routing decisions) sent to expert i
            P_i = mean routing probability for expert i
            aux_loss = num_experts * sum(f_i * P_i)

        The target value for each term is ``1 / num_experts``, giving a
        theoretical minimum aux_loss of ``1.0``.

        Args:
            router_logits: Raw routing logits ``[B*T, num_experts]``.
            expert_indices: Selected expert indices ``[B*T, top_k]``.

        Returns:
            Scalar auxiliary loss tensor.
        """
        num_tokens = router_logits.shape[0]

        # Fraction of routing decisions assigned to each expert
        expert_mask = (
            F.one_hot(expert_indices.view(-1), num_classes=self.num_experts)
            .float()
            .sum(dim=0)
        )  # [num_experts]
        expert_fraction = expert_mask / (num_tokens * self.top_k)

        # Mean routing probability for each expert
        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)  # [num_experts]

        # Load balancing loss: num_experts * sum(f_i * P_i)
        aux_loss = self.num_experts * (expert_fraction * router_prob).sum()
        return aux_loss


# --------------------------------------------------------------------------- #
# Expert Feed-Forward Network
# --------------------------------------------------------------------------- #

class ExpertFFN(nn.Module):
    """Single expert FFN: d_model -> d_ff -> d_model.

    Uses GELU activation (standard for MoE experts) with dropout.

    Args:
        d_model: Input/output dimension.
        d_ff: Hidden dimension.
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[num_tokens, d_model]``.

        Returns:
            ``[num_tokens, d_model]``.
        """
        return self.net(x)


# --------------------------------------------------------------------------- #
# MoE Layer
# --------------------------------------------------------------------------- #

class MoELayer(nn.Module):
    """Mixture of Experts layer.

    Contains ``num_experts`` expert FFNs and a learned router.  Each token is
    processed by its top-k assigned experts; outputs are weighted and
    scatter-added back into place.  A capacity factor limits the maximum
    tokens per expert (excess tokens are dropped).

    Args:
        d_model: Token embedding dimension.
        num_experts: Number of expert FFNs.
        top_k: Number of experts per token.
        expert_d_ff: Hidden dimension of each expert.  Defaults to ``d_model * 4``.
        dropout: Dropout probability.
        capacity_factor: Max tokens per expert as a fraction of the average.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: Optional[int] = None,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        expert_d_ff = expert_d_ff or d_model * 4

        # Router
        self.router = Router(d_model, num_experts, top_k)

        # Expert FFNs
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, expert_d_ff, dropout)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MoE layer.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            - **output** ``[batch, seq_len, d_model]`` -- combined expert outputs.
            - **aux_loss** scalar -- load-balancing loss from the router.
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*T, D]
        num_tokens = batch_size * seq_len

        # Route tokens to experts
        expert_indices, expert_weights, aux_loss = self.router(x_flat)
        # expert_indices: [B*T, top_k], expert_weights: [B*T, top_k]

        # Compute capacity per expert
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        capacity = max(capacity, 1)  # At least 1 token per expert

        # Accumulate output via scatter-add
        output = torch.zeros_like(x_flat)  # [B*T, D]

        for expert_idx in range(self.num_experts):
            # Find all (token_index, top_k_position) pairs routed to this expert
            token_k_pairs = (expert_indices == expert_idx).nonzero(as_tuple=False)
            # token_k_pairs: [num_assignments, 2] where columns are (token_idx, k_pos)

            if token_k_pairs.shape[0] == 0:
                continue

            # Extract token indices and their position in top-k
            token_indices = token_k_pairs[:, 0]  # [num_assignments]
            k_positions = token_k_pairs[:, 1]  # [num_assignments]

            # Capacity truncation: keep only the first `capacity` tokens
            if token_indices.shape[0] > capacity:
                token_indices = token_indices[:capacity]
                k_positions = k_positions[:capacity]

            # Gather input tokens for this expert
            expert_input = x_flat[token_indices]  # [num_assigned, D]

            # Process through the expert FFN
            expert_output = self.experts[expert_idx](expert_input)  # [num_assigned, D]

            # Get corresponding gating weights
            weights = expert_weights[token_indices, k_positions]  # [num_assigned]

            # Weighted scatter-add back to output
            output[token_indices] += expert_output * weights.unsqueeze(-1)

        output = output.view(batch_size, seq_len, d_model)
        return output, aux_loss


# --------------------------------------------------------------------------- #
# MoE Transformer Block
# --------------------------------------------------------------------------- #

class MoETransformerBlock(nn.Module):
    """Pre-norm Transformer block with MoE replacing the dense FFN.

    Architecture::

        x = x + Attention(Norm(x))
        x = x + MoE(Norm(x))

    The auxiliary loss from the MoE router is stored in ``self.aux_loss``
    so that the training loop can collect and add it to the total loss.

    Args:
        config: ``ModelConfig`` instance (must include MoE fields).
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .layers import RMSNorm
        from .flash_attention import HybridAttention

        # Attention sublayer
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = HybridAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # MoE sublayer (replaces FFN)
        self.moe_norm = RMSNorm(config.d_model, config.norm_eps)
        self.moe = MoELayer(
            d_model=config.d_model,
            num_experts=config.moe_num_experts,
            top_k=config.moe_top_k,
            expert_d_ff=config.moe_expert_d_ff,
            dropout=config.dropout,
            capacity_factor=config.moe_capacity_factor,
        )

        # Stored aux_loss for collection by the model
        self.aux_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Any] = None,
        use_cache: bool = False,
    ):
        """Forward pass through the MoE transformer block.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            mask: Optional causal mask tensor.
            kv_cache: Optional key/value cache for incremental decoding.
            use_cache: If ``True``, also return the updated KV cache.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``, or a
            ``(output, kv_cache)`` tuple when ``use_cache`` is ``True``.
        """
        # Pre-norm attention with residual
        attn_out, new_cache = self.attn(
            self.attn_norm(x), mask=mask, kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attn_out

        # Pre-norm MoE with residual
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        self.aux_loss = aux_loss

        if use_cache:
            return x, new_cache
        return x
