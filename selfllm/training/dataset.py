"""PyTorch Dataset for self-generated training data."""
import json
import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class SelfTrainingDataset(Dataset):
    """PyTorch Dataset for self-generated training data.

    Each sample is a dict with 'prompt' and 'response' text.
    __getitem__ tokenizes the combined prompt+response and creates labels
    where the prompt portion is masked with -100 (ignored in loss).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: BPETokenizer,
        max_seq_len: int = 512,
        pad_to_max: bool = True,
    ) -> None:
        """Initialize the dataset.

        Args:
            samples: List of dicts with keys 'prompt' and 'response'.
            tokenizer: BPETokenizer instance for encoding text.
            max_seq_len: Maximum sequence length (prompt + response).
            pad_to_max: If True, pad sequences to max_seq_len.
        """
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_to_max = pad_to_max
        self._preprocessed: Optional[List[Dict[str, torch.Tensor]]] = None

    def __len__(self) -> int:
        return len(self.samples)

    def _tokenize_sample(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single sample and create labels.

        Returns dict with:
            input_ids: [seq_len] tensor of token IDs
            attention_mask: [seq_len] causal mask (1 for real tokens, 0 for pad)
            labels: [seq_len] -100 for prompt tokens, actual token IDs for response
        """
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")

        # Encode prompt and response separately
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)

        # Add EOS token to response if not present
        if not response_ids or response_ids[-1] != self.tokenizer.eos_token_id:
            response_ids.append(self.tokenizer.eos_token_id)

        # Truncate if combined length exceeds max_seq_len
        combined_len = len(prompt_ids) + len(response_ids)
        if combined_len > self.max_seq_len:
            # Reserve space for prompt, truncate response
            available_for_response = self.max_seq_len - len(prompt_ids)
            if available_for_response <= 0:
                # Prompt itself is too long; truncate it and give at least 1 token to response
                prompt_ids = prompt_ids[: self.max_seq_len - 1]
                response_ids = response_ids[:1]
            else:
                response_ids = response_ids[:available_for_response]

        # Build input_ids: prompt + response
        input_ids = prompt_ids + response_ids
        seq_len = len(input_ids)

        # Labels: -100 for prompt (ignored in loss), actual tokens for response
        labels = [-100] * len(prompt_ids) + response_ids

        # Create attention mask: 1 for real tokens, 0 for padding
        attention_mask = [1] * seq_len

        # Pad to max_seq_len if requested
        if self.pad_to_max and seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            pad_id = self.tokenizer.pad_token_id
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single preprocessed sample.

        Returns:
            Dict with keys 'input_ids', 'attention_mask', 'labels'.
        """
        sample = self.samples[idx]
        return self._tokenize_sample(sample)

    def add_samples(self, new_samples: List[Dict[str, Any]]) -> None:
        """Add new self-generated samples to the dataset.

        Args:
            new_samples: List of dicts with 'prompt' and 'response' keys.
        """
        self.samples.extend(new_samples)
        # Invalidate preprocessed cache
        self._preprocessed = None
        logger.info(f"Added {len(new_samples)} samples. Total: {len(self.samples)}")

    def save(self, path: str) -> None:
        """Save dataset samples to JSON file.

        Args:
            path: Path to the output JSON file.
        """
        data = {
            "samples": self.samples,
            "max_seq_len": self.max_seq_len,
            "pad_to_max": self.pad_to_max,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(self.samples)} samples to {path}")

    @classmethod
    def load(cls, path: str, tokenizer: BPETokenizer) -> "SelfTrainingDataset":
        """Load a dataset from a JSON file.

        Args:
            path: Path to the JSON file.
            tokenizer: BPETokenizer instance.

        Returns:
            SelfTrainingDataset instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dataset = cls(
            samples=data["samples"],
            tokenizer=tokenizer,
            max_seq_len=data.get("max_seq_len", 512),
            pad_to_max=data.get("pad_to_max", True),
        )
        logger.info(f"Loaded {len(dataset.samples)} samples from {path}")
        return dataset
