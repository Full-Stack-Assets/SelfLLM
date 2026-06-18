"""Low-Rank Adaptation (LoRA) for efficient fine-tuning.

LoRA injects trainable low-rank matrices into selected linear layers of a
pre-trained transformer.  During fine-tuning the base weights stay frozen;
only the small LoRA parameters are updated, giving large memory and speed
savings.

Key formula::

    W' = W + (alpha / rank) * B @ A

where A is shaped [in_features, rank], B is [rank, out_features], and B is
initialised to zero so training starts from the base model weights.
"""

import math
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# LoRALayer
# --------------------------------------------------------------------------- #

class LoRALayer(nn.Module):
    """LoRA adaptation layer.

    Implements the low-rank update :math:`\\Delta W = (\\alpha/r) * B @ A`.

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        rank: Rank of the low-rank decomposition (``r``).
        alpha: LoRA scaling factor.  Effective scale is ``alpha / rank``.
        dropout: Dropout probability applied to the input before the low-rank
            projection.  Disabled when ``dropout <= 0``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout on the *input* side (standard LoRA practice)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialise A with Kaiming uniform; B stays zero so the initial
        # forward pass is exactly the base model.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B remains zeros — training starts from base model weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the low-rank update.

        Args:
            x: Input tensor of shape ``[..., in_features]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        # x @ A @ B * scaling
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


# --------------------------------------------------------------------------- #
# LinearWithLoRA
# --------------------------------------------------------------------------- #

class LinearWithLoRA(nn.Module):
    """Wraps an :class:`nn.Linear` with a parallel LoRA path.

    The forward pass computes::

        y = base_layer(x) + lora(x)

    The base layer weights are frozen so that only LoRA parameters receive
    gradients.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        # Freeze base layer — only LoRA parameters are trainable
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through base + LoRA path.

        Args:
            x: Input tensor of shape ``[..., in_features]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        return self.base_layer(x) + self.lora(x)


# --------------------------------------------------------------------------- #
# inject_lora
# --------------------------------------------------------------------------- #

def inject_lora(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> nn.Module:
    """Recursively find and replace target :class:`nn.Linear` layers with
    :class:`LinearWithLoRA`.

    Target module names can be partial matches — for example ``"q_proj"``
    will match ``"blocks.0.attn.q_proj"``.  The default targets (``q_proj``
    and ``v_proj``) are the most parameter-efficient choice for many tasks.

    All base model parameters are frozen so that only LoRA parameters are
    trainable after injection.

    Args:
        model: The model to modify **in-place**.
        target_modules: List of name substrings to match.  Defaults to
            ``["q_proj", "v_proj"]``.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: LoRA dropout probability.

    Returns:
        The modified ``model`` (same object, for chaining).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    def _inject_recursive(module: nn.Module, prefix: str = "") -> None:
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name

            # Skip if already a LoRA wrapper — don't recurse into it either
            if isinstance(child, LinearWithLoRA):
                continue

            if isinstance(child, nn.Linear):
                # Check if any target pattern matches this module's name
                if any(pattern in full_name for pattern in target_modules):
                    lora_layer = LinearWithLoRA(
                        child, rank=rank, alpha=alpha, dropout=dropout
                    )
                    setattr(module, name, lora_layer)
                    print(
                        f"  [LoRA] Injected into {full_name} "
                        f"-- rank={rank}, alpha={alpha}"
                    )
                continue

            # Recurse into non-Linear, non-LinearWithLoRA modules
            _inject_recursive(child, full_name)

    _inject_recursive(model)
    return model


# --------------------------------------------------------------------------- #
# merge_lora_weights
# --------------------------------------------------------------------------- #

def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base :class:`nn.Linear` layers.

    Computes :math:`W_{merged} = W + (\\alpha/r) * (A @ B).T`, updates the
    base layer weight in-place, unfreezes it, and replaces the
    :class:`LinearWithLoRA` wrapper with the base layer.

    After merging the model no longer contains LoRA paths, so the forward
    pass is a single matrix multiplication (faster inference).

    Args:
        model: Model containing :class:`LinearWithLoRA` layers.

    Returns:
        The modified ``model``.
    """

    def _merge_recursive(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, LinearWithLoRA):
                # Compute delta_W = scaling * (A @ B)
                # lora_A: [in_features, rank]
                # lora_B: [rank, out_features]
                # A @ B -> [in_features, out_features]
                # Then transpose to [out_features, in_features] for the weight.
                delta_W = (
                    child.lora.lora_A @ child.lora.lora_B
                ) * child.lora.scaling  # [in_features, out_features]
                delta_W = delta_W.T  # [out_features, in_features]

                child.base_layer.weight.data += delta_W

                # Unfreeze base layer
                for param in child.base_layer.parameters():
                    param.requires_grad = True

                # Replace wrapper with the base layer
                setattr(module, name, child.base_layer)
            else:
                _merge_recursive(child)

    _merge_recursive(model)
    return model


# --------------------------------------------------------------------------- #
# unmerge_lora_weights
# --------------------------------------------------------------------------- #

def unmerge_lora_weights(model: nn.Module) -> nn.Module:
    """Currently a no-op -- remerging requires storing original weights.

    For a future implementation: store ``W_orig`` before merging, restore
    ``W_orig`` here, and re-inject fresh :class:`LinearWithLoRA` layers.

    Args:
        model: The model (already merged or not).

    Returns:
        The unchanged ``model``.
    """
    return model


# --------------------------------------------------------------------------- #
# get_lora_parameters
# --------------------------------------------------------------------------- #

def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Return only LoRA trainable parameters.

    Useful for passing to an optimiser::

        optim = AdamW(get_lora_parameters(model), lr=1e-4)

    Args:
        model: Model containing LoRA layers.

    Returns:
        List of LoRA parameters that require gradients.
    """
    lora_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            lora_params.append(param)
    return lora_params


# --------------------------------------------------------------------------- #
# count_lora_parameters
# --------------------------------------------------------------------------- #

def count_lora_parameters(model: nn.Module) -> Dict[str, Any]:
    """Count LoRA vs. base parameters.

    Args:
        model: Model containing LoRA layers.

    Returns:
        Dictionary with keys ``lora``, ``total``, ``trainable``,
        ``frozen``, and ``lora_pct``.
    """
    lora = sum(p.numel() for p in get_lora_parameters(model))
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "lora": lora,
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "lora_pct": 100.0 * lora / total if total > 0 else 0.0,
    }


# --------------------------------------------------------------------------- #
# save_lora_weights
# --------------------------------------------------------------------------- #

def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA weights (small checkpoint).

    Args:
        model: Model containing LoRA layers.
        path: File path to save the state-dict to.  Parent directories are
            created automatically.
    """
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path)


# --------------------------------------------------------------------------- #
# load_lora_weights
# --------------------------------------------------------------------------- #

def load_lora_weights(model: nn.Module, path: str) -> nn.Module:
    """Load LoRA weights into a model.

    The model must already have LoRA layers injected (via :func:`inject_lora`).

    Args:
        model: Model with LoRA layers.
        path: Path to a LoRA checkpoint created by :func:`save_lora_weights`.

    Returns:
        The model with loaded LoRA weights.
    """
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(lora_state, strict=False)
    return model
