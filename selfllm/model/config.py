"""Model configuration dataclass for SelfLLM."""

from dataclasses import dataclass


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
