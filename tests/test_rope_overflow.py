"""Regression tests: RoPE must not crash on sequences longer than max_seq_len.

The rotary table has only ``max_seq_len`` rows; over-long prompts previously
sliced too few rows and crashed with a shape mismatch (a 500 on any prompt that
overflowed the context window). Positions now saturate at the final entry.
"""

import torch

from selfllm.model.attention import RoPEMultiHeadAttention
from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM


def test_attention_forward_past_max_seq_len():
    """A single forward over seq_len > max_seq_len rotates without crashing."""
    max_pos = 16
    attn = RoPEMultiHeadAttention(d_model=32, n_heads=4, max_seq_len=max_pos).eval()
    x = torch.randn(1, max_pos * 3, 32)  # 3x the rope table length
    out, _ = attn(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_apply_rope_saturates_length():
    """_apply_rope returns exactly seq_len positions even past the table."""
    attn = RoPEMultiHeadAttention(d_model=32, n_heads=4, max_seq_len=8)
    x = torch.randn(2, 4, 20, 8)  # seq_len 20 > max_seq_len 8
    rotated = attn._apply_rope(x, seq_len=20)
    assert rotated.shape == x.shape


def test_model_forward_and_generate_past_context():
    """End-to-end: full-model forward and generate past max_seq_len don't crash."""
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                      d_ff=64, max_seq_len=16, dropout=0.0)
    model = SelfImprovingLLM(cfg).eval()

    # Forward a prompt 2.5x the context window.
    long_ids = torch.randint(0, 64, (1, 40))
    out = model(long_ids)
    assert out["logits"].shape == (1, 40, 64)
    assert torch.isfinite(out["logits"]).all()

    # Greedy-generate starting from a prompt already near the window, past it.
    prompt = torch.randint(0, 64, (1, 14))
    gen = model.generate(prompt, max_new_tokens=10, temperature=0.0)
    assert gen["sequences"].shape[1] == 14 + 10
