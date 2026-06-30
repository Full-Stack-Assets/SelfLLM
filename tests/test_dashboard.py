"""Tests for the Gradio dashboard helpers."""

from __future__ import annotations

import torch

from selfllm import dashboard


class _TinyTokenizer:
    eos_token_id = 1

    def encode(self, text: str) -> list[int]:
        self.last_prompt = text
        return [10, 20]

    def decode(self, token_ids: list[int]) -> str:
        if token_ids == [30, 31]:
            return "Hello from SelfLLM"
        if token_ids == [10, 20, 30, 31]:
            return f"{self.last_prompt} Hello from SelfLLM"
        return "decoded"


class _TinyModel:
    def __init__(self) -> None:
        self.param = torch.nn.Parameter(torch.zeros(1))
        self.config = type("Config", (), {"max_seq_len": 128})()

    def parameters(self):
        yield self.param

    def eval(self):
        return self

    def generate(self, prompt_ids, **kwargs):
        self.last_kwargs = kwargs
        return {"sequences": torch.tensor([[10, 20, 30, 31]])}


def test_format_chat_prompt_includes_system_history_and_next_assistant_turn():
    prompt = dashboard._format_chat_prompt(
        "What now?",
        [("Hi", "Hello")],
        "Be concise.",
    )

    assert prompt == (
        "System: Be concise.\n"
        "User: Hi\n"
        "Assistant: Hello\n"
        "User: What now?\n"
        "Assistant:"
    )


def test_messages_to_turns_converts_gradio_messages():
    turns = dashboard._messages_to_turns(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Next"},
            {"role": "assistant", "content": "Done"},
        ]
    )

    assert turns == [("Hi", "Hello"), ("Next", "Done")]


def test_format_generation_stats_renders_telemetry():
    text = dashboard._format_generation_stats(
        {
            "new_tokens": 8,
            "elapsed_s": 0.5,
            "tokens_per_second": 16.0,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 12,
        }
    )

    assert "`8` new tokens" in text
    assert "`16.0` tok/s" in text
    assert "temperature `0.7`" in text


def test_format_transcript_exports_json_messages():
    transcript = dashboard._format_transcript(
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    )

    assert '"role": "user"' in transcript
    assert '"content": "Hello"' in transcript


def test_handle_chat_reports_missing_model():
    state = dashboard.DashboardState()

    history, textbox, stats = dashboard._handle_chat(
        "Hello",
        [],
        "",
        0.8,
        0.92,
        16,
        1.0,
        state,
    )

    assert textbox == ""
    assert "No model loaded" in stats
    assert history == [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": "Error: No model loaded. Load a model in Settings or start "
            "with `selfllm dashboard --model-path ... --tokenizer-path ...`.",
        },
    ]


def test_handle_chat_generates_assistant_reply():
    model = _TinyModel()
    tokenizer = _TinyTokenizer()
    state = dashboard.DashboardState(model=model, tokenizer=tokenizer)

    history, textbox, stats = dashboard._handle_chat(
        "Say hello",
        [],
        "Be friendly.",
        0.7,
        0.9,
        12,
        1.05,
        state,
    )

    assert textbox == ""
    assert "`2` new tokens" in stats
    assert history == [
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello from SelfLLM"},
    ]
    assert tokenizer.last_prompt.endswith("User: Say hello\nAssistant:")
    assert model.last_kwargs["max_new_tokens"] == 12
    assert model.last_kwargs["temperature"] == 0.7
    assert model.last_kwargs["top_p"] == 0.9
    assert model.last_kwargs["repetition_penalty"] == 1.05
    assert model.last_kwargs["stop_token_id"] == tokenizer.eos_token_id
