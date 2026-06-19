"""Tests for sliding-window KV-cache eviction (bounded long-context decode)."""

import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM


def _cache(length, sinks_dim=False):
    """A one-layer cache [(k, v)] of seq length `length`; k[...,0] encodes index."""
    k = torch.zeros(1, 1, length, 2)
    k[0, 0, :, 0] = torch.arange(length, dtype=torch.float32)
    v = k.clone()
    return [(k, v)]


def test_evict_trims_to_budget():
    past = _cache(10)
    out = SelfImprovingLLM._evict_kv_cache(past, sinks=2, window=3)
    k = out[0][0]
    assert k.shape[2] == 5  # sinks(2) + window(3)
    # Kept indices: first 2 sinks (0, 1) + last 3 (7, 8, 9).
    assert k[0, 0, :, 0].tolist() == [0.0, 1.0, 7.0, 8.0, 9.0]


def test_evict_no_sinks():
    out = SelfImprovingLLM._evict_kv_cache(_cache(10), sinks=0, window=4)
    assert out[0][0].shape[2] == 4
    assert out[0][0][0, 0, :, 0].tolist() == [6.0, 7.0, 8.0, 9.0]


def test_evict_under_budget_untouched():
    past = _cache(4)
    out = SelfImprovingLLM._evict_kv_cache(past, sinks=2, window=3)  # budget 5 > 4
    assert out[0][0].shape[2] == 4
    assert torch.equal(out[0][0], past[0][0])


def _windowed_model(window=4, sinks=1, max_seq_len=64):
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=2, n_heads=4, d_ff=64,
                      max_seq_len=max_seq_len, dropout=0.0,
                      sliding_window=window, attention_sinks=sinks)
    return SelfImprovingLLM(cfg).eval(), window, sinks


def test_generate_bounds_cache():
    """During windowed generation the KV cache never exceeds sinks + window."""
    model, window, sinks = _windowed_model(window=4, sinks=1)
    captured = []

    def spy(past, s, w):
        out = SelfImprovingLLM._evict_kv_cache(past, s, w)
        captured.append(out[0][0].shape[2])
        return out

    model._evict_kv_cache = spy  # instance shadow; called as self._evict_kv_cache(...)
    prompt = torch.randint(0, 64, (1, 3))
    out = model.generate(prompt, max_new_tokens=30, temperature=0.0)

    assert captured, "eviction should have run"
    assert max(captured) <= sinks + window
    assert out["sequences"].shape[1] == 3 + 30


def test_generate_windowed_finite():
    """Windowed generation past the window runs and stays finite."""
    model, _, _ = _windowed_model(window=4, sinks=1)
    out = model.generate(torch.randint(0, 64, (1, 5)), max_new_tokens=20,
                         temperature=0.8, top_p=0.9)
    assert out["sequences"].shape[1] == 5 + 20
    assert torch.isfinite(out["scores"]).all()


def test_streaming_attention_matches_full():
    """Attention-time (streaming) RoPE, fed incrementally with no eviction,
    matches a single full forward (cache-relative == absolute positions)."""
    from selfllm.model.attention import RoPEMultiHeadAttention

    torch.manual_seed(0)
    attn = RoPEMultiHeadAttention(d_model=32, n_heads=4, max_seq_len=64).eval()
    x = torch.randn(1, 6, 32)
    with torch.no_grad():
        full, _ = attn(x)
        cache, outs = None, []
        for t in range(6):
            o, cache = attn(x[:, t:t + 1], kv_cache=cache, streaming=True)
            outs.append(o)
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(full, stream, atol=1e-5)


def test_streaming_generation_unbounded():
    """Windowed generation far past max_seq_len works (positions stay
    cache-relative, so they never saturate the rotary table)."""
    model, window, sinks = _windowed_model(window=4, sinks=1, max_seq_len=16)
    captured = []

    def spy(past, s, w):
        out = SelfImprovingLLM._evict_kv_cache(past, s, w)
        captured.append(out[0][0].shape[2])
        return out

    model._evict_kv_cache = spy
    # Generate ~3x the context window of 16.
    out = model.generate(torch.randint(0, 64, (1, 4)), max_new_tokens=44,
                         temperature=0.0)
    assert out["sequences"].shape[1] == 4 + 44
    assert torch.isfinite(out["scores"]).all()
    # Cache stayed bounded the whole time, well under max_seq_len.
    assert max(captured) <= sinks + window
    assert max(captured) < model.config.max_seq_len
