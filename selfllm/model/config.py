"""Model configuration dataclass for SelfLLM."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for the SelfImprovingLLM model.

    Attributes:
        vocab_size: Vocabulary size (default: 32000).
        d_model: Embedding and hidden state dimension (default: 512).
        n_layers: Number of transformer layers (default: 8).
        n_heads: Number of attention heads (default: 8).
        d_ff: Feed-forward hidden dimension (default: 2048).
        max_seq_len: Maximum sequence length (default: 512).
        dropout: Dropout rate applied throughout the model (default: 0.1).
        tie_weights: Whether to tie input/output embedding weights (default: True).
        use_rope: Use Rotary Position Embedding (default: True).
        use_swiglu: Use SwiGLU activation in FFN (default: True).
        norm_eps: Epsilon for numerical stability in layer norms (default: 1e-6).
        grad_checkpoint: Enable gradient checkpointing to save memory (default: False).
        use_moe: Use Mixture of Experts instead of dense FFN (default: False).
        moe_num_experts: Number of expert FFNs in MoE layers (default: 8).
        moe_top_k: Number of experts each token routes to (default: 2).
        moe_expert_d_ff: Hidden dim of each expert. None means d_model * 4 (default: None).
        moe_capacity_factor: Max tokens per expert as fraction of average (default: 1.25).
        moe_aux_loss_weight: Weight for the load-balancing auxiliary loss (default: 0.01).
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    tie_weights: bool = True
    use_rope: bool = True
    use_swiglu: bool = True
    norm_eps: float = 1e-6
    grad_checkpoint: bool = False
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_expert_d_ff: Optional[int] = None
    moe_capacity_factor: float = 1.25
    moe_aux_loss_weight: float = 0.01

    # Long-context attention. When ``sliding_window`` is set, each token attends
    # only to the most recent ``sliding_window`` tokens, plus the first
    # ``attention_sinks`` "sink" tokens -- breaking the quadratic full-context
    # attention for long sequences.
    sliding_window: Optional[int] = None
    attention_sinks: int = 0
