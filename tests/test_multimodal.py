"""Unit tests for VisionEncoder and MultimodalLLM.

Tests cover:
- VisionEncoder output shapes
- Patch embedding correctness
- MultimodalLLM forward pass (with and without images)
- Text-only mode (no images)
- Generation methods
- Gradient flow through vision encoder
"""

import pytest
import torch

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.vision import MultimodalLLM, VisionEncoder


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def tiny_llm_config() -> ModelConfig:
    """Tiny LLM config for fast tests."""
    return ModelConfig(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        max_seq_len=512,
        dropout=0.0,
        tie_weights=True,
    )


@pytest.fixture
def tiny_llm(tiny_llm_config: ModelConfig) -> SelfImprovingLLM:
    """Instantiated tiny LLM."""
    return SelfImprovingLLM(tiny_llm_config)


@pytest.fixture
def tiny_vision_encoder(tiny_llm_config: ModelConfig) -> VisionEncoder:
    """Tiny vision encoder matching the LLM's d_model."""
    return VisionEncoder(
        image_size=224,
        patch_size=16,
        in_channels=3,
        d_model=tiny_llm_config.d_model,
        n_layers=2,
        n_heads=4,
        d_ff=128,
        dropout=0.0,
    )


@pytest.fixture
def tiny_multimodal(
    tiny_llm: SelfImprovingLLM,
    tiny_vision_encoder: VisionEncoder,
) -> MultimodalLLM:
    """Instantiated tiny multimodal model."""
    return MultimodalLLM(tiny_llm, tiny_vision_encoder)


@pytest.fixture
def device() -> str:
    """Compute device for tests."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dummy_image(device: str) -> torch.Tensor:
    """Create a dummy image tensor."""
    return torch.randn(2, 3, 224, 224, device=device)


@pytest.fixture
def dummy_text_ids(device: str) -> torch.Tensor:
    """Create dummy text token IDs."""
    return torch.randint(0, 256, (2, 16), device=device)


# --------------------------------------------------------------------------- #
# 1. VisionEncoder output shape
# --------------------------------------------------------------------------- #


class TestVisionEncoder:
    """Tests for the VisionEncoder (ViT)."""

    def test_vision_encoder_shape(
        self,
        tiny_vision_encoder: VisionEncoder,
        device: str,
    ) -> None:
        """VisionEncoder should output [B, 196, D] for 224x224 / 16 patches."""
        tiny_vision_encoder = tiny_vision_encoder.to(device)
        tiny_vision_encoder.eval()

        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224, device=device)

        with torch.no_grad():
            outputs = tiny_vision_encoder(images)

        num_patches = (224 // 16) ** 2  # 196
        d_model = tiny_vision_encoder.d_model

        assert outputs.shape == (batch_size, num_patches, d_model), (
            f"Expected shape ({batch_size}, {num_patches}, {d_model}), "
            f"got {outputs.shape}"
        )

    def test_vision_encoder_patch_embedding(
        self,
        tiny_vision_encoder: VisionEncoder,
        device: str,
    ) -> None:
        """Number of patches should equal (image_size // patch_size)^2."""
        tiny_vision_encoder = tiny_vision_encoder.to(device)
        tiny_vision_encoder.eval()

        images = torch.randn(1, 3, 224, 224, device=device)

        with torch.no_grad():
            outputs = tiny_vision_encoder(images)

        expected_num_patches = (224 // 16) ** 2  # 196
        assert outputs.shape[1] == expected_num_patches, (
            f"Expected {expected_num_patches} patches, got {outputs.shape[1]}"
        )

    def test_vision_encoder_different_batch_sizes(
        self,
        tiny_vision_encoder: VisionEncoder,
        device: str,
    ) -> None:
        """VisionEncoder should handle varying batch sizes."""
        tiny_vision_encoder = tiny_vision_encoder.to(device)
        tiny_vision_encoder.eval()

        for batch_size in [1, 2, 8]:
            images = torch.randn(batch_size, 3, 224, 224, device=device)
            with torch.no_grad():
                outputs = tiny_vision_encoder(images)
            assert outputs.shape[0] == batch_size
            assert outputs.shape[1] == 196
            assert outputs.shape[2] == tiny_vision_encoder.d_model

    def test_vision_encoder_parameter_count(
        self,
        tiny_vision_encoder: VisionEncoder,
    ) -> None:
        """VisionEncoder should report its parameter count."""
        num_params = tiny_vision_encoder.count_parameters()
        assert num_params > 0, "VisionEncoder should have parameters"

    def test_vision_encoder_gradients(
        self,
        tiny_vision_encoder: VisionEncoder,
        device: str,
    ) -> None:
        """Gradients should flow through the vision encoder."""
        tiny_vision_encoder = tiny_vision_encoder.to(device)
        tiny_vision_encoder.train()

        images = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        outputs = tiny_vision_encoder(images)

        # Backward through a scalar
        loss = outputs.sum()
        loss.backward()

        assert images.grad is not None, "Gradients should flow to input images"
        assert images.grad.shape == images.shape

        # Check that transformer block parameters have gradients
        for block in tiny_vision_encoder.blocks:
            for p in block.parameters():
                assert p.grad is not None, (
                    "TransformerBlock parameters should have gradients"
                )


# --------------------------------------------------------------------------- #
# 2. MultimodalLLM forward pass
# --------------------------------------------------------------------------- #


class TestMultimodalLLM:
    """Tests for the MultimodalLLM wrapper."""

    def test_multimodal_forward_shape(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Forward pass should return logits of shape [B, text_seq_len, vocab_size]."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.eval()

        batch_size = 2
        text_seq_len = 16
        text_ids = torch.randint(0, 256, (batch_size, text_seq_len), device=device)
        images = torch.randn(batch_size, 3, 224, 224, device=device)

        with torch.no_grad():
            outputs = tiny_multimodal(text_ids, images=images)

        assert "logits" in outputs
        logits = outputs["logits"]
        assert logits.shape == (batch_size, text_seq_len, 256), (
            f"Expected logits shape ({batch_size}, {text_seq_len}, 256), "
            f"got {logits.shape}"
        )

    def test_multimodal_forward_no_loss_without_targets(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Forward pass without targets should not include loss."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.eval()

        text_ids = torch.randint(0, 256, (2, 8), device=device)
        images = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            outputs = tiny_multimodal(text_ids, images=images)

        assert "loss" not in outputs

    def test_multimodal_forward_with_loss(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Forward pass with targets should include a scalar loss."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.train()

        text_ids = torch.randint(0, 256, (2, 8), device=device)
        images = torch.randn(2, 3, 224, 224, device=device)
        targets = torch.randint(0, 256, (2, 8), device=device)

        outputs = tiny_multimodal(text_ids, images=images, targets=targets)

        assert "loss" in outputs
        assert outputs["loss"].dim() == 0, "Loss should be a scalar"
        assert outputs["loss"] >= 0, "Cross-entropy loss should be non-negative"

    def test_multimodal_with_image(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Multimodal forward with images should include image information."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.eval()

        text_ids = torch.randint(0, 256, (1, 10), device=device)

        # Forward with images
        images = torch.randn(1, 3, 224, 224, device=device)
        with torch.no_grad():
            outputs_with = tiny_multimodal(text_ids, images=images)

        # Forward without images (same text)
        with torch.no_grad():
            outputs_without = tiny_multimodal(text_ids, images=None)

        # The logits should differ because image information is included
        assert "logits" in outputs_with
        assert "logits" in outputs_without

        # They should not be identical (image presence changes the output)
        assert not torch.allclose(
            outputs_with["logits"], outputs_without["logits"]
        ), (
            "Outputs should differ when images are present vs. absent"
        )

    def test_multimodal_text_only(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Multimodal forward should work without images (text-only mode)."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.eval()

        batch_size = 2
        text_seq_len = 12
        text_ids = torch.randint(0, 256, (batch_size, text_seq_len), device=device)

        with torch.no_grad():
            outputs = tiny_multimodal(text_ids, images=None)

        assert "logits" in outputs
        logits = outputs["logits"]
        assert logits.shape == (batch_size, text_seq_len, 256)

    def test_multimodal_text_only_matches_llm(
        self,
        tiny_multimodal: MultimodalLLM,
        tiny_llm: SelfImprovingLLM,
        device: str,
    ) -> None:
        """Text-only multimodal forward should match base LLM forward."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_llm = tiny_llm.to(device)
        tiny_multimodal.eval()
        tiny_llm.eval()

        text_ids = torch.randint(0, 256, (2, 10), device=device)

        with torch.no_grad():
            mm_outputs = tiny_multimodal(text_ids, images=None)
            llm_outputs = tiny_llm(text_ids)

        # Logits should be very close (small numerical differences allowed)
        assert torch.allclose(
            mm_outputs["logits"], llm_outputs["logits"], atol=1e-5
        ), (
            "Text-only multimodal output should match base LLM output"
        )

    def test_multimodal_gradient_flow(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Gradients should flow through the vision encoder in end-to-end training."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.train()

        text_ids = torch.randint(0, 256, (2, 8), device=device)
        images = torch.randn(2, 3, 224, 224, device=device, requires_grad=True)
        targets = torch.randint(0, 256, (2, 8), device=device)

        outputs = tiny_multimodal(text_ids, images=images, targets=targets)
        loss = outputs["loss"]
        loss.backward()

        # Gradient should flow to image input
        assert images.grad is not None, (
            "Gradients should flow back to image input"
        )
        assert images.grad.shape == images.shape

        # Gradient should flow through vision encoder
        for block in tiny_multimodal.vision.blocks:
            for p in block.parameters():
                assert p.grad is not None, (
                    "Vision transformer parameters should have gradients"
                )

        # Gradient should also flow through LLM
        for block in tiny_multimodal.llm.blocks:
            for p in block.parameters():
                assert p.grad is not None, (
                    "LLM parameters should have gradients"
                )


# --------------------------------------------------------------------------- #
# 3. Generation methods
# --------------------------------------------------------------------------- #


class TestGeneration:
    """Tests for generate_image_caption and answer_visual_question."""

    def test_generate_image_caption_returns_string(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """generate_image_caption should return a string."""
        tiny_multimodal = tiny_multimodal.to(device)

        image = torch.randn(1, 3, 224, 224, device=device)
        caption = tiny_multimodal.generate_image_caption(
            image, max_length=10, temperature=1.0
        )

        assert isinstance(caption, str), (
            f"Expected str, got {type(caption)}"
        )
        assert len(caption) > 0, "Caption should not be empty"

    def test_generate_image_caption_single_image(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """generate_image_caption should work with 3D image tensor."""
        tiny_multimodal = tiny_multimodal.to(device)

        # 3D tensor [C, H, W] — should add batch dim internally
        image = torch.randn(3, 224, 224, device=device)
        caption = tiny_multimodal.generate_image_caption(
            image, max_length=5, temperature=1.0
        )

        assert isinstance(caption, str)
        assert len(caption) > 0

    def test_answer_visual_question_returns_string(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """answer_visual_question should return a string."""
        tiny_multimodal = tiny_multimodal.to(device)

        image = torch.randn(1, 3, 224, 224, device=device)
        question_ids = torch.randint(0, 256, (10,), device=device)

        answer = tiny_multimodal.answer_visual_question(
            image, question_ids, max_length=10, temperature=1.0
        )

        assert isinstance(answer, str), (
            f"Expected str, got {type(answer)}"
        )
        assert len(answer) > 0, "Answer should not be empty"

    def test_answer_visual_question_2d_question(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """answer_visual_question should work with 2D question tensor."""
        tiny_multimodal = tiny_multimodal.to(device)

        image = torch.randn(1, 3, 224, 224, device=device)
        question_ids = torch.randint(0, 256, (1, 10), device=device)

        answer = tiny_multimodal.answer_visual_question(
            image, question_ids, max_length=5, temperature=1.0
        )

        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_answer_visual_question_3d_image(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """answer_visual_question should work with 3D image tensor."""
        tiny_multimodal = tiny_multimodal.to(device)

        image = torch.randn(3, 224, 224, device=device)
        question_ids = torch.randint(0, 256, (8,), device=device)

        answer = tiny_multimodal.answer_visual_question(
            image, question_ids, max_length=5, temperature=1.0
        )

        assert isinstance(answer, str)
        assert len(answer) > 0


# --------------------------------------------------------------------------- #
# 4. Integration / edge cases
# --------------------------------------------------------------------------- #


class TestIntegration:
    """Integration and edge-case tests."""

    def test_d_model_mismatch_raises_error(
        self,
        tiny_llm: SelfImprovingLLM,
    ) -> None:
        """Creating MultimodalLLM with mismatched d_model should raise ValueError."""
        # Vision encoder with different d_model than LLM
        vision = VisionEncoder(
            image_size=224,
            patch_size=16,
            in_channels=3,
            d_model=128,  # Different from LLM's d_model=64
            n_layers=2,
            n_heads=4,
            d_ff=128,
            dropout=0.0,
        )

        with pytest.raises(ValueError, match="d_model"):
            MultimodalLLM(tiny_llm, vision)

    def test_parameter_counts(
        self,
        tiny_multimodal: MultimodalLLM,
    ) -> None:
        """Parameter count methods should return positive values."""
        total = tiny_multimodal.count_parameters()
        vision = tiny_multimodal.count_vision_parameters()
        llm = tiny_multimodal.count_llm_parameters()

        assert total > 0
        assert vision > 0
        assert llm > 0
        assert total == vision + llm, (
            "Total parameters should equal vision + LLM parameters"
        )

    def test_special_token_ids(
        self,
        tiny_multimodal: MultimodalLLM,
    ) -> None:
        """Special token IDs should be set correctly."""
        vocab_size = tiny_multimodal.llm.config.vocab_size

        assert tiny_multimodal.image_start_token_id == vocab_size
        assert tiny_multimodal.image_end_token_id == vocab_size + 1

    def test_vision_encoder_deterministic(
        self,
        tiny_vision_encoder: VisionEncoder,
        device: str,
    ) -> None:
        """VisionEncoder should produce deterministic outputs in eval mode."""
        tiny_vision_encoder = tiny_vision_encoder.to(device)
        tiny_vision_encoder.eval()

        images = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            out1 = tiny_vision_encoder(images)
            out2 = tiny_vision_encoder(images)

        assert torch.allclose(out1, out2), (
            "Same input should produce same output in eval mode"
        )

    def test_multimodal_batch_independence(
        self,
        tiny_multimodal: MultimodalLLM,
        device: str,
    ) -> None:
        """Different batch items should not interfere with each other."""
        tiny_multimodal = tiny_multimodal.to(device)
        tiny_multimodal.eval()

        # Process each item in the batch individually
        text_ids = torch.randint(0, 256, (2, 8), device=device)
        images = torch.randn(2, 3, 224, 224, device=device)

        with torch.no_grad():
            batch_outputs = tiny_multimodal(text_ids, images=images)

        # Process first item alone
        with torch.no_grad():
            single_outputs = tiny_multimodal(
                text_ids[0:1], images=images[0:1]
            )

        # First batch item should match single item
        assert torch.allclose(
            batch_outputs["logits"][0], single_outputs["logits"][0], atol=1e-5
        )
