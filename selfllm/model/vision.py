"""Vision Transformer encoder + Multimodal LLM for image+text understanding.

Architecture:
    - VisionEncoder: lightweight ViT that processes images into patch embeddings
    - MultimodalLLM: wraps the base LLM + vision encoder, concatenating
      image patch embeddings with text token embeddings before the LLM backbone.
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, TransformerBlock
from .model import SelfImprovingLLM


# --------------------------------------------------------------------------- #
# Vision Encoder (ViT)
# --------------------------------------------------------------------------- #


class VisionEncoder(nn.Module):
    """Vision Transformer encoder for multimodal understanding.

    Processes images into patch embeddings that the LLM can understand.
    Uses a lightweight ViT (smaller than CLIP, designed to be fused with LLM).

    Args:
        image_size: Input image height/width in pixels (default: 224).
        patch_size: Patch height/width in pixels (default: 16).
        in_channels: Number of input channels, 3 for RGB (default: 3).
        d_model: Embedding dimension — **must match the LLM's** ``d_model``
            so that patch embeddings can be fed directly into the LLM.
        n_layers: Number of ViT transformer layers (default: 6).
        n_heads: Number of attention heads in each ViT layer (default: 8).
        d_ff: Feed-forward hidden dimension (default: 2048).
        dropout: Dropout rate (default: 0.1).
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model

        # ------------------------------------------------------------------ #
        # Patch embedding via Conv2d — each patch -> d_model vector
        # ------------------------------------------------------------------ #
        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ------------------------------------------------------------------ #
        # CLS token + learnable position embeddings
        # ------------------------------------------------------------------ #
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # +1 position for the CLS token
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)
        )

        # ------------------------------------------------------------------ #
        # Transformer blocks (share the same architecture as the LLM)
        # ------------------------------------------------------------------ #
        # Build a ModelConfig for the vision transformer blocks.
        # max_seq_len must cover num_patches + 1 (CLS token).
        vit_config = ModelConfig(
            vocab_size=1,  # not used by TransformerBlock
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ff=d_ff,
            max_seq_len=self.num_patches + 1,
            dropout=dropout,
            tie_weights=False,
            use_rope=True,
            use_swiglu=True,
            norm_eps=1e-6,
            grad_checkpoint=False,
        )
        self.blocks = nn.ModuleList(
            [TransformerBlock(vit_config) for _ in range(n_layers)]
        )

        # Final layer norm
        self.norm = RMSNorm(d_model, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------ #
        # Initialise embeddings
        # ------------------------------------------------------------------ #
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images into patch embeddings.

        Args:
            images: Input images ``[batch, channels, height, width]``.

        Returns:
            Patch embeddings ``[batch, num_patches, d_model]``.
            The CLS token is **excluded** — only spatial patch tokens are
            returned so they can be interleaved with text token embeddings.
        """
        batch_size = images.shape[0]

        # Patch embedding: [B, C, H, W] -> [B, D, H/P, W/P] -> [B, num_patches, D]
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]

        # Prepend CLS token: [B, num_patches+1, D]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.dropout(x)

        # Pass through ViT transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return all patch tokens (excluding CLS)
        return x[:, 1:, :]  # [B, num_patches, D]

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Multimodal LLM
# --------------------------------------------------------------------------- #


class MultimodalLLM(nn.Module):
    """Multimodal LLM that processes both images and text.

    Architecture:
        1. Vision encoder processes images -> patch embeddings.
        2. Text token IDs -> token embeddings via the LLM's embedding table.
        3. Concatenate ``[image_patch_embeddings, text_token_embeddings]``.
        4. Process the combined sequence through the LLM transformer blocks.
        5. Compute logits and loss **only on the text portion**.

    Args:
        llm: The base ``SelfImprovingLLM`` language model.
        vision_encoder: The ``VisionEncoder`` for processing images.
        image_start_token_id: Token ID that marks the start of an image
            region in the text sequence (default: ``vocab_size``).
        image_end_token_id: Token ID that marks the end of an image region
            (default: ``vocab_size + 1``).
    """

    def __init__(
        self,
        llm: SelfImprovingLLM,
        vision_encoder: VisionEncoder,
        image_start_token_id: Optional[int] = None,
        image_end_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.llm = llm
        self.vision = vision_encoder

        # Special tokens for image boundaries — placed *after* the existing
        # vocabulary so they don't collide with real tokens.
        self.image_start_token_id = (
            image_start_token_id or llm.config.vocab_size
        )
        self.image_end_token_id = (
            image_end_token_id or llm.config.vocab_size + 1
        )

        # Verify that d_model dimensions match
        if vision_encoder.d_model != llm.config.d_model:
            raise ValueError(
                f"Vision encoder d_model ({vision_encoder.d_model}) must match "
                f"LLM d_model ({llm.config.d_model})"
            )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        text_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the multimodal model.

        When ``images`` is provided, image patch embeddings are prepended to
        the text token embeddings before being fed into the LLM transformer.
        Logits and loss are computed **only on the text portion** of the
        output sequence.

        Args:
            text_ids: Text token IDs ``[batch, text_seq_len]``.
            images: Optional image tensor ``[batch, 3, 224, 224]``.
            targets: Optional target token IDs for loss computation
                ``[batch, text_seq_len]``.

        Returns:
            Dictionary with:
                - ``logits``: ``[batch, text_seq_len, vocab_size]``
                - ``loss``: scalar cross-entropy (only when ``targets`` given)
        """
        batch_size, text_seq_len = text_ids.shape
        device = text_ids.device

        # Get text token embeddings
        text_embeds = self.llm.token_embedding(text_ids)  # [B, T_text, D]

        if images is not None:
            # Get image patch embeddings
            image_embeds = self.vision(images)  # [B, num_patches, D]

            # Concatenate: [image_patches, text_tokens]
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            num_image_tokens = image_embeds.shape[1]
        else:
            combined_embeds = text_embeds
            num_image_tokens = 0

        combined_seq_len = combined_embeds.shape[1]

        # Build causal mask for the combined sequence
        mask = torch.triu(
            torch.ones(
                combined_seq_len, combined_seq_len, device=device, dtype=torch.bool
            ),
            diagonal=1,
        )

        # Pass through LLM transformer blocks
        x = combined_embeds
        x = self.llm.dropout(x)
        for block in self.llm.blocks:
            if self.llm.config.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, mask)
            else:
                x = block(x, mask=mask)

        # Final norm
        hidden_states = self.llm.norm(x)

        # Only compute logits for the *text* portion
        text_hidden = hidden_states[:, num_image_tokens:, :]  # [B, T_text, D]
        logits = self.llm.lm_head(text_hidden)  # [B, T_text, vocab_size]

        result: Dict[str, torch.Tensor] = {"logits": logits}

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.llm.config.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    # ------------------------------------------------------------------ #
    # Generation helpers
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_image_caption(
        self,
        image: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        prompt_token_ids: Optional[torch.Tensor] = None,
    ) -> str:
        """Generate a caption for an image.

        Encodes the image through the vision encoder once, then autoregressively
        generates text tokens conditioned on the image patch embeddings.

        Args:
            image: Input image ``[1, 3, 224, 224]`` or ``[3, 224, 224]``.
            max_length: Maximum number of text tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Nucleus (top-p) filtering parameter.
            prompt_token_ids: Optional prompt token IDs to prepend to the
                generation. If not provided, a default "caption:" prompt
                is used (as raw token IDs from the base vocab).

        Returns:
            Generated caption string (token IDs space-joined if no tokenizer
            is available; callers with a tokenizer should decode the returned
            sequence themselves).
        """
        self.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, 224, 224]

        batch_size = image.shape[0]
        device = image.device

        # Encode image -> patch embeddings (done once)
        image_embeds = self.vision(image)  # [B, num_patches, D]
        num_image_tokens = image_embeds.shape[1]

        # Prepare text prompt — must use valid in-vocab token IDs
        if prompt_token_ids is None:
            # Start with a single <eos> token as the generation seed
            prompt_token_ids = torch.tensor(
                [[1]], device=device, dtype=torch.long  # <eos> token ID
            )
        elif prompt_token_ids.dim() == 1:
            prompt_token_ids = prompt_token_ids.unsqueeze(0)

        # Validate prompt token IDs are within vocab range
        vocab_size = self.llm.config.vocab_size
        if prompt_token_ids.max() >= vocab_size or prompt_token_ids.min() < 0:
            raise ValueError(
                f"prompt_token_ids must be in range [0, {vocab_size}), "
                f"got min={prompt_token_ids.min().item()}, "
                f"max={prompt_token_ids.max().item()}"
            )

        # Get text embeddings for the prompt
        text_embeds = self.llm.token_embedding(prompt_token_ids)  # [B, T_prompt, D]

        # Initial combined sequence: [image_patches, prompt_tokens]
        combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        generated_token_ids = prompt_token_ids.clone()

        for _step in range(max_length):
            combined_seq_len = combined_embeds.shape[1]

            # Causal mask
            mask = torch.triu(
                torch.ones(
                    combined_seq_len, combined_seq_len, device=device, dtype=torch.bool
                ),
                diagonal=1,
            )

            # Forward through LLM blocks
            x = combined_embeds
            for block in self.llm.blocks:
                x = block(x, mask=mask)
            x = self.llm.norm(x)

            # Extract logits for the last text position
            text_logits = self.llm.lm_head(x[:, -1, :])  # [B, vocab_size]

            # Temperature scaling
            if temperature > 0:
                logits = text_logits / temperature

                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = (
                        sorted_indices_to_remove[:, :-1].clone()
                    )
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
            else:
                # Greedy
                next_token = torch.argmax(text_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated_token_ids = torch.cat([generated_token_ids, next_token], dim=1)

            # Get embedding for the new token and append to combined sequence
            next_embed = self.llm.token_embedding(next_token)  # [B, 1, D]
            combined_embeds = torch.cat([combined_embeds, next_embed], dim=1)

        # Return as space-joined token IDs (caller can decode with tokenizer)
        token_list = generated_token_ids[0].cpu().tolist()
        return " ".join(str(t) for t in token_list)

    @torch.no_grad()
    def answer_visual_question(
        self,
        image: torch.Tensor,
        question_token_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Answer a question about an image.

        Encodes the image once, prepends the question tokens, then
        autoregressively generates the answer.

        Args:
            image: Input image ``[1, 3, 224, 224]`` or ``[3, 224, 224]``.
            question_token_ids: Token IDs of the question ``[seq_len]`` or
                ``[1, seq_len]``.
            max_length: Maximum number of answer tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Nucleus (top-p) filtering parameter.

        Returns:
            Generated answer string (space-joined token IDs).
        """
        self.eval()

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)  # [1, 3, 224, 224]
        if question_token_ids.dim() == 1:
            question_token_ids = question_token_ids.unsqueeze(0)  # [1, T_q]

        # Encode image
        image_embeds = self.vision(image)  # [B, num_patches, D]

        # Get question embeddings
        question_embeds = self.llm.token_embedding(question_token_ids)  # [B, T_q, D]

        # Combined: [image_patches, question_tokens]
        combined_embeds = torch.cat([image_embeds, question_embeds], dim=1)
        generated_token_ids = question_token_ids.clone()

        for _step in range(max_length):
            combined_seq_len = combined_embeds.shape[1]

            # Causal mask
            mask = torch.triu(
                torch.ones(
                    combined_seq_len, combined_seq_len, device=image.device, dtype=torch.bool
                ),
                diagonal=1,
            )

            # Forward through LLM blocks
            x = combined_embeds
            for block in self.llm.blocks:
                x = block(x, mask=mask)
            x = self.llm.norm(x)

            # Logits for last position
            text_logits = self.llm.lm_head(x[:, -1, :])  # [B, vocab_size]

            # Temperature scaling + sampling
            if temperature > 0:
                logits = text_logits / temperature

                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = (
                        sorted_indices_to_remove[:, :-1].clone()
                    )
                    sorted_indices_to_remove[:, 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(text_logits, dim=-1, keepdim=True)

            generated_token_ids = torch.cat([generated_token_ids, next_token], dim=1)

            next_embed = self.llm.token_embedding(next_token)
            combined_embeds = torch.cat([combined_embeds, next_embed], dim=1)

        token_list = generated_token_ids[0].cpu().tolist()
        return " ".join(str(t) for t in token_list)

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Return total trainable parameters (LLM + vision)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_vision_parameters(self) -> int:
        """Return trainable parameters in the vision encoder only."""
        return sum(
            p.numel() for p in self.vision.parameters() if p.requires_grad
        )

    def count_llm_parameters(self) -> int:
        """Return trainable parameters in the LLM only."""
        return sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
