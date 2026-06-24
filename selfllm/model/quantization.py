"""Quantization (GPTQ-style 4-bit and AWQ activation-aware) for SelfLLM.

Provides:
    - QuantizedLinear: 4-bit / 8-bit quantized linear layer
    - quantize_model: replace all nn.Linear layers with QuantizedLinear
    - save_quantized / load_quantized: efficient serialization
    - AWQQuantizer: activation-aware weight quantization
"""

import math
import warnings
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# QuantizedLinear — grouped 4-bit / 8-bit linear layer
# --------------------------------------------------------------------------- #

class QuantizedLinear(nn.Module):
    """Linear layer with grouped weight quantization.

    Weights are stored as packed integers (two 4-bit values per ``uint8`` byte,
    or one 8-bit value per byte) and dequantized on-the-fly during the forward
    pass.  Each group of ``group_size`` consecutive (flattened) weights shares
    its own scale and zero-point.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bits: Bit-width per weight (``4`` or ``8``).
        group_size: Number of weights that share a scale / zero-point.
        bias: Whether to include a bias term (rarely used for LLM linear layers).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()

        if bits not in (4, 8):
            raise ValueError(f"Only 4-bit and 8-bit quantization are supported, got bits={bits}")
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if (in_features * out_features) % group_size != 0:
            # Pad the weight matrix so the total element count is divisible.
            warnings.warn(
                f"Weight matrix size ({in_features * out_features}) is not divisible "
                f"by group_size ({group_size}).  Quantization will pad internally.",
                stacklevel=2,
            )

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.num_elements = in_features * out_features

        # Number of groups after padding to group_size boundary
        self.num_groups = math.ceil(self.num_elements / group_size)
        padded_elements = self.num_groups * group_size

        # Packed weight storage
        if bits == 4:
            # Two 4-bit weights per uint8 byte
            self.pack_dim = padded_elements // 2
        else:
            # One 8-bit weight per uint8 byte
            self.pack_dim = padded_elements

        self.register_buffer(
            "qweight",
            torch.zeros(self.pack_dim, dtype=torch.uint8),
        )
        self.register_buffer(
            "scales",
            torch.zeros(self.num_groups, dtype=torch.float32),
        )
        self.register_buffer(
            "zeros",
            torch.zeros(self.num_groups, dtype=torch.float32),
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=torch.float32),
            )
        else:
            self.register_parameter("bias", None)

        # Lazily-computed cache of the dequantized float weight (see
        # ``dequantize_cached``).  Not a buffer/parameter so it is excluded from
        # the state dict; invalidated whenever quantized params change.
        self._dequant_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------ #
    # Quantization helpers
    # ------------------------------------------------------------------ #

    def _quantize_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a flattened float32 weight tensor to integers.

        Returns the *quantized integer values* (before packing) of shape
        ``[padded_elements]`` and stores ``scales`` / ``zeros``.
        """
        device = weight.device
        flat = weight.detach().view(-1).contiguous().float()

        # Pad to group_size boundary
        padded_len = self.num_groups * self.group_size
        if flat.numel() < padded_len:
            padding = torch.zeros(padded_len - flat.numel(), device=device, dtype=torch.float32)
            flat = torch.cat([flat, padding])

        # Grouped min-max quantization
        flat_grouped = flat.view(self.num_groups, self.group_size)

        w_min = flat_grouped.min(dim=1, keepdim=True).values
        w_max = flat_grouped.max(dim=1, keepdim=True).values

        qmax = 2**self.bits - 1
        # Avoid division by zero
        scale = (w_max - w_min) / qmax
        scale = scale.squeeze(1)
        zero_point = w_min.squeeze(1)

        # Quantize: q = round((w - zp) / scale)
        q = torch.round((flat_grouped - w_min) / (w_max - w_min).clamp_min(1e-8) * qmax)
        q = q.clamp(0, qmax).to(torch.int32)

        # Store quantization params
        self.scales.copy_(scale.detach().cpu().float())
        self.zeros.copy_(zero_point.detach().cpu().float())

        return q.view(-1)

    def _pack_weights(self, q: torch.Tensor) -> None:
        """Pack quantized integers into the ``qweight`` uint8 buffer."""
        # Quantized params are changing; any cached dequantized weight is stale.
        self._invalidate_weight_cache()
        if self.bits == 8:
            self.qweight.copy_(q.to(torch.uint8).cpu())
        elif self.bits == 4:
            # Pack two 4-bit values per byte: high nibble = first weight, low nibble = second
            q_even = q[0::2]
            q_odd = q[1::2]
            # Ensure equal length
            min_len = min(q_even.numel(), q_odd.numel())
            packed = (q_even[:min_len] << 4 | q_odd[:min_len]).to(torch.uint8)
            self.qweight[: packed.numel()].copy_(packed.cpu())

    def _unpack_weights(self) -> torch.Tensor:
        """Unpack ``qweight`` uint8 buffer to integer tensor.

        Returns quantized integer values of shape ``[num_elements]`` (unpadded).
        """
        if self.bits == 8:
            return self.qweight.to(torch.int32)
        else:
            # Unpack 4-bit: high nibble and low nibble
            high = (self.qweight >> 4) & 0xF
            low = self.qweight & 0xF
            unpacked = torch.stack([high, low], dim=1).view(-1).to(torch.int32)
            return unpacked[: self.num_elements]

    # ------------------------------------------------------------------ #
    # Dequantize
    # ------------------------------------------------------------------ #

    def dequantize(self) -> torch.Tensor:
        """Dequantize packed weights back to ``float32``.

        Returns:
            Weight tensor of shape ``[out_features, in_features]``.
        """
        # Unpack to integers
        q = self._unpack_weights()  # [num_elements]

        # Determine which group each element belongs to
        group_indices = torch.arange(q.numel(), device=q.device) // self.group_size
        group_indices = group_indices.clamp(0, self.num_groups - 1)

        scales = self.scales.to(q.device)[group_indices]
        zeros = self.zeros.to(q.device)[group_indices]

        # Dequantize: w = (q / (2**self.bits - 1)) * scale + zero_point  ... wait, standard is:
        # w = (q - zp) * scale, where zp is the zero-point in quantized space
        # We stored zero_point as w_min, and scale = (w_max - w_min) / qmax
        # q = round((w - w_min) / scale)
        # So w = q * scale + w_min = q * scale + zero_point
        weight = q.float() * scales + zeros

        # Reshape to original matrix shape
        weight = weight[: self.num_elements].view(self.out_features, self.in_features)
        return weight.contiguous()

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def _invalidate_weight_cache(self) -> None:
        """Drop any cached dequantized weight (call when quantized params change)."""
        self._dequant_cache = None

    def dequantize_cached(self) -> torch.Tensor:
        """Return the dequantized weight, computing it once and caching it.

        Note: true low-memory / faster inference requires int8 matmul kernels
        that are not available in pure PyTorch, so dequantizing to float is
        unavoidable here.  To avoid redoing identical work on every forward
        pass, we cache the dequantized weight and reuse it.  The cache is
        invalidated whenever the quantized parameters change (see
        ``_invalidate_weight_cache``).  Numerical results are unchanged versus
        calling :meth:`dequantize` directly.
        """
        cached = getattr(self, "_dequant_cache", None)
        if cached is not None:
            return cached
        weight = self.dequantize()
        self._dequant_cache = weight
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear transform using the (cached) dequantized weights."""
        weight = self.dequantize_cached()
        return F.linear(x, weight, self.bias)

    # ------------------------------------------------------------------ #
    # Factory
    # ------------------------------------------------------------------ #

    @classmethod
    def quantize_from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        group_size: int = 128,
    ) -> "QuantizedLinear":
        """Create a ``QuantizedLinear`` by quantizing an existing ``nn.Linear``.

        Args:
            linear: The linear layer to quantize.
            bits: Target bit-width (``4`` or ``8``).
            group_size: Group size for grouped quantization.

        Returns:
            A ``QuantizedLinear`` instance with quantized weights.
        """
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            group_size=group_size,
            bias=linear.bias is not None,
        )

        # Quantize the weight matrix
        q = qlinear._quantize_weights(linear.weight.data)
        qlinear._pack_weights(q)

        if linear.bias is not None:
            qlinear.bias.copy_(linear.bias.data.detach().cpu().float())

        return qlinear

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"num_groups={self.num_groups}"
        )


# --------------------------------------------------------------------------- #
# Model-level quantization
# --------------------------------------------------------------------------- #

def _count_linear_layers(module: nn.Module) -> int:
    """Count the number of ``nn.Linear`` layers in a module."""
    return sum(1 for m in module.modules() if isinstance(m, nn.Linear))


def _get_model_size_mb(module: nn.Module) -> float:
    """Estimate model size in MB from parameter buffers."""
    total_bytes = sum(
        p.numel() * p.element_size() for p in module.parameters()
    )
    total_bytes += sum(
        b.numel() * b.element_size() for b in module.buffers()
    )
    return total_bytes / (1024 * 1024)


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    group_size: int = 128,
) -> nn.Module:
    """Recursively replace all ``nn.Linear`` layers with ``QuantizedLinear``.

    Args:
        model: The model to quantize (modified in-place).
        bits: Target bit-width (``4`` or ``8``).
        group_size: Group size for grouped quantization.

    Returns:
        The same model instance with all linear layers quantized.
    """
    orig_size = _get_model_size_mb(model)
    num_replaced = 0

    def _replace(module: nn.Module) -> None:
        nonlocal num_replaced
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                qlinear = QuantizedLinear.quantize_from_linear(
                    child, bits=bits, group_size=group_size
                )
                setattr(module, name, qlinear)
                num_replaced += 1
            else:
                _replace(child)

    _replace(model)
    new_size = _get_model_size_mb(model)
    savings_pct = (orig_size - new_size) / orig_size * 100 if orig_size > 0 else 0.0

    print(
        f"[quantize_model] Replaced {num_replaced} Linear layers with QuantizedLinear "
        f"(bits={bits}, group_size={group_size})."
    )
    print(
        f"[quantize_model] Model size: {orig_size:.2f} MB -> {new_size:.2f} MB "
        f"({savings_pct:.1f}% reduction)"
    )

    return model


# --------------------------------------------------------------------------- #
# Save / Load quantized models
# --------------------------------------------------------------------------- #

def save_quantized(model: nn.Module, path: str) -> None:
    """Save a quantized model efficiently.

    Only saves ``qweight``, ``scales``, ``zeros``, and metadata — *not* full
    ``float32`` weights.  Metadata (``in_features``, ``out_features``,
    ``bits``, ``group_size``, ``bias``) is stored alongside each layer.

    Args:
        model: Model containing ``QuantizedLinear`` layers.
        path: Destination file path (``.pt`` or ``.pth``).
    """
    state: Dict[str, Any] = {
        "_quantized": True,
        "_format_version": 1,
    }

    # Collect names of all QuantizedLinear layers (to skip their sub-keys)
    quantized_prefixes: set = set()
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            quantized_prefixes.add(name)

    # Save compact representation for each QuantizedLinear
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLinear):
            state[name] = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bits": module.bits,
                "group_size": module.group_size,
                "qweight": module.qweight.cpu(),
                "scales": module.scales.cpu(),
                "zeros": module.zeros.cpu(),
                "has_bias": module.bias is not None,
            }
            if module.bias is not None:
                state[name]["bias"] = module.bias.cpu()
            if hasattr(module, "awq_scale"):
                state[name]["awq_scale"] = module.awq_scale.cpu()

    # Save regular state_dict entries that are NOT part of QuantizedLinear layers
    for key, tensor in model.state_dict().items():
        # Skip if this key belongs to a QuantizedLinear layer
        is_quantized = any(
            key == prefix or key.startswith(prefix + ".")
            for prefix in quantized_prefixes
        )
        if not is_quantized:
            state[key] = tensor.cpu()

    torch.save(state, path)
    print(f"[save_quantized] Saved quantized model to {path}")


def load_quantized(
    path: str,
    model: Optional[nn.Module] = None,
) -> nn.Module:
    """Load a quantized model from a saved file.

    Args:
        path: Path to the saved quantized model.
        model: Optional model instance to load weights into.  If provided,
            ``QuantizedLinear`` layers are reconstructed and swapped in.
            If ``None``, the saved state dict is returned as a flat dict.

    Returns:
        The loaded model (or the state dict if no model was provided).
    """
    # Security: load with weights_only=True so a malicious checkpoint cannot
    # execute arbitrary code during unpickling.  The payload written by
    # ``save_quantized`` is composed only of tensors and primitive containers
    # (dict/list/int/float/bool/str), all of which are permitted by the safe
    # weights_only loader.
    state = torch.load(path, map_location="cpu", weights_only=True)

    if not state.pop("_quantized", False):
        warnings.warn("File does not appear to be a quantized model save.", stacklevel=2)

    state.pop("_format_version", None)

    if model is None:
        return state  # type: ignore[return-value]

    # Reconstruct QuantizedLinear layers
    quantized_layers: Dict[str, Dict[str, Any]] = {}
    regular_state: Dict[str, torch.Tensor] = {}

    for key, value in state.items():
        if isinstance(value, dict) and "qweight" in value:
            quantized_layers[key] = value
        else:
            regular_state[key] = value

    # Replace nn.Linear with QuantizedLinear in the model
    for name, qdata in quantized_layers.items():
        # Navigate to the parent module
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]

        qlinear = QuantizedLinear(
            in_features=qdata["in_features"],
            out_features=qdata["out_features"],
            bits=qdata["bits"],
            group_size=qdata["group_size"],
            bias=qdata.get("has_bias", False),
        )
        qlinear.qweight.copy_(qdata["qweight"])
        qlinear.scales.copy_(qdata["scales"])
        qlinear.zeros.copy_(qdata["zeros"])
        if qdata.get("has_bias") and "bias" in qdata:
            qlinear.bias.copy_(qdata["bias"])
        if "awq_scale" in qdata:
            qlinear.register_buffer("awq_scale", qdata["awq_scale"])

        setattr(parent, child_name, qlinear)

    # Build expected state keys for reconstructed QuantizedLinear layers
    # so we can suppress warnings about them being "missing" from load_state_dict
    expected_qlinear_keys: set = set()
    for prefix in quantized_layers.keys():
        for suffix in ("qweight", "scales", "zeros", "bias", "awq_scale"):
            expected_qlinear_keys.add(f"{prefix}.{suffix}")

    # Load remaining regular parameters (only for non-QuantizedLinear modules)
    if regular_state:
        # Filter out keys that belong to modules that were replaced with QuantizedLinear
        # (their original state dict keys like blocks.0.attn.q_proj.weight no longer exist)
        valid_state = {}
        for key, tensor in regular_state.items():
            is_replaced = any(
                key.startswith(prefix + ".") for prefix in quantized_layers.keys()
            )
            if not is_replaced:
                valid_state[key] = tensor

        if valid_state:
            missing, unexpected = model.load_state_dict(valid_state, strict=False)
            # Only warn about truly unexpected missing keys
            real_missing = [k for k in missing if k not in expected_qlinear_keys]
            if real_missing:
                warnings.warn(f"Missing keys when loading regular state: {real_missing}", stacklevel=2)

    return model


# --------------------------------------------------------------------------- #
# AWQ — Activation-aware Weight Quantization
# --------------------------------------------------------------------------- #

class AWQQuantizer:
    """Activation-aware weight quantizer.

    AWQ scales weights channel-wise based on observed activation magnitudes
    before applying quantization.  Channels that correspond to larger
    activation magnitudes receive higher precision (via scaling), which
    reduces the effective quantization error for important features.

    Algorithm:
        1. Run calibration data → collect per-channel activation magnitudes.
        2. Derive channel-wise scaling factors from activations.
        3. Scale weights: ``W' = W * s`` where ``s`` is the activation scale.
        4. Quantize scaled weights (better preserves important weights).
        5. Store quantized weights + inverse scale for dequantization.
    """

    def __init__(self) -> None:
        self.activation_stats: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Run calibration data and collect per-channel activation magnitudes.

        Args:
            model: The model to calibrate.
            calibration_data: Input tensor ``[batch, seq_len]`` (token IDs) or
                ``[batch, seq_len, d_model]`` (embeddings).

        Returns:
            Dictionary mapping module names to per-channel activation
            magnitude tensors.
        """
        self.activation_stats.clear()
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(name: str):
            def hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
                # output shape: [batch, seq_len, out_features] or [batch, seq_len, d_model]
                # Collect per-channel (last dim) mean absolute activation
                if isinstance(output, torch.Tensor):
                    # Average over batch and sequence dimensions
                    mag = output.abs().mean(dim=(0, 1))  # [out_features] or [d_model]
                    if name in self.activation_stats:
                        self.activation_stats[name] = (
                            self.activation_stats[name] + mag
                        ) / 2
                    else:
                        self.activation_stats[name] = mag

            return hook

        # Register hooks on all Linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                handles.append(module.register_forward_hook(_make_hook(name)))

        # Run calibration
        model.eval()
        try:
            _ = model(calibration_data)
        except (RuntimeError, ValueError, TypeError, IndexError) as exc:
            # These are the expected failure modes when the calibration input
            # doesn't match the model's expected signature/shape (e.g. passing
            # token ids where embeddings are expected).  We surface a visible
            # warning rather than silently swallowing every possible error.
            warnings.warn(
                f"AWQ calibration forward pass failed ({type(exc).__name__}: {exc}). "
                f"Activation statistics may be incomplete.",
                stacklevel=2,
            )
        finally:
            for h in handles:
                h.remove()

        if not self.activation_stats:
            raise RuntimeError(
                "AWQ calibration collected no activation statistics. The "
                "calibration forward pass likely failed or the model has no "
                "nn.Linear layers; check that ``calibration_data`` matches the "
                "model's expected input."
            )

        return dict(self.activation_stats)

    # ------------------------------------------------------------------ #
    # Quantize with activation awareness
    # ------------------------------------------------------------------ #

    def quantize(
        self,
        model: nn.Module,
        bits: int = 4,
        group_size: int = 128,
    ) -> nn.Module:
        """Apply AWQ quantization to a model.

        Before standard quantization, each linear layer's weight matrix is
        scaled channel-wise (row-wise for ``W`` in ``y = x @ W.T``) by the
        activation magnitudes collected during calibration.

        Args:
            model: Model to quantize (modified in-place).
            bits: Target bit-width.
            group_size: Group size for grouped quantization.

        Returns:
            The quantized model.
        """
        if not self.activation_stats:
            raise RuntimeError(
                "No activation statistics collected. Run ``calibrate()`` first."
            )

        orig_size = _get_model_size_mb(model)
        num_replaced = 0

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Get activation magnitudes for this layer
            mag = self.activation_stats.get(name)
            if mag is None or mag.numel() != module.out_features:
                # Fall back to uniform scaling
                mag = torch.ones(module.out_features, device=module.weight.device)

            # Normalize magnitudes to get scaling factors
            # Channels with larger activations get larger scale factors
            mag = mag.to(module.weight.device)
            scale = mag.clamp_min(1e-6)  # Avoid division by zero

            # Scale weights channel-wise: W' = W * s.unsqueeze(1)
            # Each output channel (row of W) is scaled by its activation magnitude
            scaled_weight = module.weight.data * scale.unsqueeze(1)

            # Create a temporary linear with scaled weights for quantization
            tmp_linear = nn.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                device=module.weight.device,
            )
            tmp_linear.weight.data.copy_(scaled_weight)
            if module.bias is not None:
                tmp_linear.bias.data.copy_(module.bias.data)

            # Quantize the scaled weights
            qlinear = QuantizedLinear.quantize_from_linear(
                tmp_linear, bits=bits, group_size=group_size
            )

            # Store the inverse scale for proper dequantization
            # When we dequantize: W_deq = dequantize(qweight) * (1 / s)
            # But since we scaled before quantization, we need to unscale after
            qlinear.register_buffer(
                "awq_scale",
                (1.0 / scale).detach().cpu().float(),
            )

            # Replace the original layer
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], qlinear)
            num_replaced += 1

        new_size = _get_model_size_mb(model)
        savings_pct = (orig_size - new_size) / orig_size * 100 if orig_size > 0 else 0.0

        print(
            f"[AWQQuantizer] Replaced {num_replaced} Linear layers with AWQ-quantized "
            f"QuantizedLinear (bits={bits}, group_size={group_size})."
        )
        print(
            f"[AWQQuantizer] Model size: {orig_size:.2f} MB -> {new_size:.2f} MB "
            f"({savings_pct:.1f}% reduction)"
        )

        return model


# --------------------------------------------------------------------------- #
# Convenience: QuantizedLinear with AWQ-aware forward
# --------------------------------------------------------------------------- #

def apply_awq_forward_patch(model: nn.Module) -> None:
    """Patch all ``QuantizedLinear`` layers to apply AWQ inverse scaling.

    If a ``QuantizedLinear`` has an ``awq_scale`` buffer, this replaces its
    ``forward`` method with one that multiplies the dequantized weights by
    the inverse activation scale before applying the linear transform.
    """
    for module in model.modules():
        if isinstance(module, QuantizedLinear) and hasattr(module, "awq_scale"):
            # Create a closure that captures the module and its awq_scale
            _patch_awq_forward(module)


def _patch_awq_forward(qlinear: QuantizedLinear) -> None:
    """Replace forward on a single QuantizedLinear to include AWQ un-scaling."""
    awq_scale = qlinear.awq_scale  # [out_features]

    def awq_forward(x: torch.Tensor) -> torch.Tensor:
        weight = qlinear.dequantize()  # [out_features, in_features]
        # Apply inverse activation scale per output channel
        scale_device = awq_scale.to(weight.device)
        weight = weight * scale_device.unsqueeze(1)
        return F.linear(x, weight, qlinear.bias)

    qlinear.forward = awq_forward  # type: ignore[method-assign]
