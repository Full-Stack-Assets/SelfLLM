"""Tests for the Bradley-Terry reward-model trainer."""

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer
from selfllm.training.ppo_trainer import RewardModel
from selfllm.training.reward_trainer import RewardModelTrainer


@pytest.fixture
def reward_trainer():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=64, dropout=0.0)
    model = SelfImprovingLLM(cfg)
    tok = BPETokenizer(vocab_size=256)
    tok.train([
        "good helpful correct answer", "bad wrong unhelpful reply",
        "the chosen response is preferred", "the rejected response is worse",
    ])
    rm = RewardModel(model)
    return RewardModelTrainer(rm, tok, learning_rate=1e-3, batch_size=4, device="cpu")


def _pairs():
    return [
        {"prompt": "Q: help me. ", "chosen": "good helpful correct answer",
         "rejected": "bad wrong unhelpful reply"},
        {"prompt": "Q: explain. ", "chosen": "the chosen response is preferred",
         "rejected": "the rejected response is worse"},
        {"prompt": "Q: write. ", "chosen": "good correct preferred answer",
         "rejected": "bad worse wrong reply"},
        {"prompt": "Q: do it. ", "chosen": "helpful chosen correct",
         "rejected": "unhelpful rejected wrong"},
    ]


def test_empty_pairs_zero_metrics(reward_trainer):
    assert reward_trainer.train_step([]) == {
        "loss": 0.0, "accuracy": 0.0, "reward_margin": 0.0
    }


def test_metrics_shape(reward_trainer):
    m = reward_trainer.train_step(_pairs(), num_epochs=1)
    assert set(m) == {"loss", "accuracy", "reward_margin"}
    assert all(isinstance(v, float) for v in m.values())


def test_learns_to_rank(reward_trainer):
    pairs = _pairs()
    before = reward_trainer.evaluate(pairs)
    reward_trainer.train_step(pairs, num_epochs=60)
    after = reward_trainer.evaluate(pairs)
    # After training, the reward model ranks every chosen above its rejected,
    # with a positive margin -- the whole point of the Bradley-Terry objective.
    assert after["accuracy"] == 1.0
    assert after["reward_margin"] > 0.0
    assert after["reward_margin"] > before["reward_margin"]


def test_score_returns_float(reward_trainer):
    s = reward_trainer.score("good helpful correct answer")
    assert isinstance(s, float)


def test_save_load_roundtrip(reward_trainer, tmp_path):
    pairs = _pairs()
    reward_trainer.train_step(pairs, num_epochs=10)
    path = str(tmp_path / "reward.pt")
    reward_trainer.save(path)
    score_before = reward_trainer.score(pairs[0]["prompt"] + pairs[0]["chosen"])

    # Fresh trainer wrapping a new reward model, load the saved weights.
    cfg = ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                      d_ff=128, max_seq_len=64, dropout=0.0)
    rm2 = RewardModel(SelfImprovingLLM(cfg))
    rt2 = RewardModelTrainer(rm2, reward_trainer.tokenizer, device="cpu")
    rt2.load(path)
    score_after = rt2.score(pairs[0]["prompt"] + pairs[0]["chosen"])
    assert score_before == pytest.approx(score_after, abs=1e-4)
