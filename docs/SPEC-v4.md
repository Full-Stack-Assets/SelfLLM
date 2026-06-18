# SPEC-v4.md — Frontier Research Features for SelfLLM

## 1. Mixture of Experts (MoE) (`selfllm/model/moe.py`)

Switch Transformer-style sparse MoE. Instead of one dense FFN per layer, use N smaller "expert" FFNs + a router. Each token activates only top-k experts.

### Key Concepts
- **Router**: Linear layer that computes routing scores [batch*seq, num_experts]
- **Top-k gating**: Each token selects k experts with highest scores
- **Load balancing**: Auxiliary loss ensures experts are equally utilized
- **Capacity factor**: Maximum tokens per expert (prevents overflow)

### Implementation

```python
class Router(nn.Module):
    """Routes tokens to experts."""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: [batch*seq_len, d_model]
        
        Returns:
            expert_indices: [batch*seq_len, top_k] — which expert each token goes to
            expert_weights: [batch*seq_len, top_k] — gating weights
            aux_loss: scalar — load balancing loss
        """
        # Compute routing scores
        router_logits = self.gate(x)  # [B*T, num_experts]
        
        # Softmax to get weights
        weights = F.softmax(router_logits, dim=-1)  # [B*T, num_experts]
        
        # Top-k experts per token
        expert_weights, expert_indices = torch.topk(weights, self.top_k, dim=-1)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        # Load balancing auxiliary loss
        # Encourage uniform expert utilization
        aux_loss = self._compute_load_balancing_loss(router_logits, expert_indices)
        
        return expert_indices, expert_weights, aux_loss
    
    def _compute_load_balancing_loss(
        self, router_logits: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        f_i = fraction of tokens routed to expert i (target: 1/num_experts)
        P_i = mean routing probability for expert i
        aux_loss = num_experts * sum(f_i * P_i)
        """
        num_tokens = router_logits.shape[0]
        
        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(
            expert_indices.view(-1), num_classes=self.num_experts
        ).float().sum(dim=0) / (num_tokens * self.top_k)
        
        # Mean routing probability
        router_prob = F.softmax(router_logits, dim=-1).mean(dim=0)
        
        aux_loss = self.num_experts * (expert_mask * router_prob).sum()
        return aux_loss


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Contains N expert FFNs + 1 router. Each token is processed by top-k experts.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expert_d_ff: Optional[int] = None,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        expert_d_ff = expert_d_ff or d_model * 4
        
        # Router
        self.router = Router(d_model, num_experts, top_k)
        
        # Experts (each is a small FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [batch, seq_len, d_model]
        
        Returns:
            output: [batch, seq_len, d_model]
            aux_loss: scalar
        """
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [B*T, D]
        
        # Route tokens to experts
        expert_indices, expert_weights, aux_loss = self.router(x_flat)
        # expert_indices: [B*T, top_k], expert_weights: [B*T, top_k]
        
        # Compute capacity per expert
        capacity = int(self.capacity_factor * (batch_size * seq_len) / self.num_experts)
        
        # Process each expert's assigned tokens
        output = torch.zeros_like(x_flat)  # [B*T, D]
        
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            mask = (expert_indices == expert_idx).any(dim=-1)  # [B*T]
            token_indices = mask.nonzero(as_tuple=True)[0]
            
            if len(token_indices) == 0:
                continue
            
            # Capacity truncation
            if len(token_indices) > capacity:
                token_indices = token_indices[:capacity]
            
            # Get tokens
            expert_input = x_flat[token_indices]  # [num_tokens, D]
            
            # Process
            expert_output = self.experts[expert_idx](expert_input)  # [num_tokens, D]
            
            # Weighted combination
            # Find which position in top-k this expert is
            expert_pos = (expert_indices[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            weights = expert_weights[token_indices, expert_pos]  # [num_tokens]
            
            output[token_indices] += expert_output * weights.unsqueeze(-1)
        
        output = output.view(batch_size, seq_len, d_model)
        return output, aux_loss


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE replacing the FFN."""
    
    def __init__(self, config: "ModelConfig", moe_config: Optional[Dict] = None):
        super().__init__()
        from .layers import RMSNorm, HybridAttention
        
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = HybridAttention(
            config.d_model, config.n_heads, config.max_seq_len, config.dropout
        )
        
        self.moe_norm = RMSNorm(config.d_model, config.norm_eps)
        
        moe_config = moe_config or {}
        self.moe = MoELayer(
            d_model=config.d_model,
            num_experts=moe_config.get("num_experts", 8),
            top_k=moe_config.get("top_k", 2),
            dropout=config.dropout,
        )
        
        self.aux_loss = 0.0
    
    def forward(self, x, mask=None, kv_cache=None, is_prefill=True):
        # Attention sublayer
        attn_out, _ = self.attn(self.attn_norm(x), mask, kv_cache, is_prefill)
        x = x + attn_out
        
        # MoE sublayer
        moe_out, aux_loss = self.moe(self.moe_norm(x))
        x = x + moe_out
        self.aux_loss = aux_loss
        
        return x
```

### ModelConfig additions:
```python
@dataclass
class ModelConfig:
    # ... existing fields ...
    use_moe: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_aux_loss_weight: float = 0.01  # Weight for load balancing loss
```

### Integration:
- Add `use_moe` flag to `ModelConfig`
- In `SelfImprovingLLM.__init__`, use `MoETransformerBlock` when `use_moe=True`
- Sum auxiliary losses across all MoE layers into total loss
- `get_350m_config()` should have `use_moe=True, moe_num_experts=8, moe_top_k=2`

---

## 2. Multimodal Vision + Text (`selfllm/model/vision.py`)

Add a Vision Transformer (ViT) encoder that projects image patches into the LLM's embedding space. The LLM then processes image + text tokens together.

### Implementation

```python
class VisionEncoder(nn.Module):
    """
    Vision Transformer encoder for multimodal understanding.
    
    Processes images into patch embeddings that the LLM can understand.
    Uses a lightweight ViT (smaller than CLIP, designed to be fused with LLM).
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,  # Must match LLM d_model
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.d_model = d_model
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embedding for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, d_model)  # +1 for CLS token
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer layers
        from .layers import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [batch, channels, height, width]
        
        Returns: [batch, num_patches, d_model] — patch embeddings
        """
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, D]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Return all patch tokens (excluding CLS)
        return x[:, 1:, :]  # [B, num_patches, D]


class MultimodalLLM(nn.Module):
    """
    Multimodal LLM that can process both images and text.
    
    Architecture:
    1. Vision encoder processes images -> patch embeddings
    2. Text tokenizer -> token embeddings
    3. Concatenate [image_patches, text_tokens]
    4. Process with standard LLM
    5. Decode only text portion
    """
    
    def __init__(
        self,
        llm: SelfImprovingLLM,
        vision_encoder: VisionEncoder,
    ):
        super().__init__()
        self.llm = llm
        self.vision = vision_encoder
        
        # Special tokens for image boundaries
        self.image_start_token_id = llm.config.vocab_size  # Add to vocab
        self.image_end_token_id = llm.config.vocab_size + 1
    
    def forward(
        self,
        text_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        text_ids: [batch, text_seq_len]
        images: [batch, 3, 224, 224] (optional)
        targets: [batch, text_seq_len] (optional, for loss)
        
        Returns: {"logits": [...], "loss": float}
        """
        # Get text embeddings
        text_embeds = self.llm.token_embedding(text_ids)  # [B, T, D]
        
        if images is not None:
            # Get image patch embeddings
            image_embeds = self.vision(images)  # [B, num_patches, D]
            
            # Concatenate: [image_patches, text_tokens]
            combined_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        else:
            combined_embeds = text_embeds
        
        # Pass through LLM transformer
        x = combined_embeds
        for block in self.llm.blocks:
            x = block(x)
        x = self.llm.norm(x)
        
        # Only compute logits for text portion
        logits = self.llm.lm_head(x[:, -text_ids.shape[1]:, :])
        
        output = {"logits": logits}
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            output["loss"] = loss
        
        return output
    
    def generate_image_caption(
        self,
        image: torch.Tensor,
        max_length: int = 50,
    ) -> str:
        """Generate a caption for an image."""
        ...
    
    def answer_visual_question(
        self,
        image: torch.Tensor,
        question: str,
        max_length: int = 50,
    ) -> str:
        """Answer a question about an image."""
        ...
```

---

## 3. Long Context Memory (`selfllm/model/long_context.py`)

Break the max_seq_len barrier using:
1. **Sliding Window Attention** — Each token only attends to the last W tokens
2. **Attention Sinks** — Keep first K tokens in KV cache (they act as "sinks")
3. **StreamingLLM** — Combine sliding window + attention sinks for infinite context
4. **RAG Retrieval** — External vector DB for long-document retrieval

### Implementation

```python
class SlidingWindowAttention(nn.Module):
    """Attention with sliding window. Each token only attends to last W tokens."""
    
    def __init__(self, d_model: int, n_heads: int, window_size: int = 1024):
        super().__init__()
        self.window_size = window_size
        # ... standard attention init ...
    
    def forward(self, x, mask=None):
        # Standard attention but mask everything outside window
        seq_len = x.shape[1]
        # Create sliding window mask
        window_mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size + 1)
            window_mask[i, start:i+1] = 0
        # Apply mask and compute attention
        ...


class StreamingAttention(nn.Module):
    """
    StreamingLLM attention: attention sinks + sliding window.
    
    Key insight: First few tokens (attention sinks) must ALWAYS be kept in KV cache.
    Without them, attention scores explode when KV cache is truncated.
    
    Algorithm:
    1. Always keep first K tokens (attention sinks) in KV cache
    2. Plus the most recent W tokens (sliding window)
    3. Total effective context: K + W tokens (but can process infinite input)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        window_size: int = 1024,
        num_sink_tokens: int = 4,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_sink_tokens = num_sink_tokens
        # ... standard attention init ...
    
    def forward(self, x, kv_cache=None):
        # If KV cache exceeds window_size, evict middle tokens
        # Keep: sink tokens (first num_sink_tokens) + recent window_size tokens
        ...


class RAGRetriever:
    """Retrieval-Augmented Generation with vector database."""
    
    def __init__(self, embedding_dim: int, index_path: Optional[str] = None):
        """
        Args:
            embedding_dim: Dimension of embeddings
            index_path: Path to saved FAISS index (optional)
        """
        try:
            import faiss
            self.use_faiss = True
        except ImportError:
            self.use_faiss = False
        
        self.embedding_dim = embedding_dim
        self.documents = []
        
        if self.use_faiss:
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity
        else:
            self.index = None
    
    def add_documents(self, documents: List[str], embeddings: torch.Tensor) -> None:
        """Add documents to the retrieval index."""
        self.documents.extend(documents)
        if self.use_faiss:
            self.index.add(embeddings.numpy())
        else:
            # Simple brute-force fallback
            if not hasattr(self, '_embeddings'):
                self._embeddings = embeddings
            else:
                self._embeddings = torch.cat([self._embeddings, embeddings], dim=0)
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant documents."""
        if self.use_faiss:
            scores, indices = self.index.search(query_embedding.numpy(), k)
            return [
                {"document": self.documents[i], "score": float(scores[0][j])}
                for j, i in enumerate(indices[0])
            ]
        else:
            # Brute-force cosine similarity
            similarities = (self._embeddings @ query_embedding.T).squeeze()
            top_k = torch.topk(similarities, k)
            return [
                {"document": self.documents[i], "score": float(top_k.values[j])}
                for j, i in enumerate(top_k.indices)
            ]
```

---

## 4. Constitutional AI + PPO (`selfllm/training/ppo_trainer.py`, `selfllm/training/constitutional_ai.py`)

### Constitutional AI
Model critiques its own outputs according to a set of principles ("constitution"), then revises them. Train on the revised outputs.

```python
class ConstitutionalAI:
    """
    Constitutional AI: Self-critique and revision.
    
    Process:
    1. Generate initial response
    2. Critique response against constitutional principles
     3. Generate revised response based on critique
    4. Train model to prefer revised over initial
    
    Constitution examples:
    - "Is the response helpful, honest, and harmless?"
    - "Does the response contain any harmful content?"
    - "Is the response factually accurate?"
    """
    
    CONSTITUTION = [
        "Please critique the following response. Is it helpful, honest, and harmless?",
        "Does this response contain any biases, stereotypes, or harmful assumptions?",
        "Is this response factually accurate to the best of your knowledge?",
        "Could this response be misinterpreted or used to cause harm?",
        "Is the response appropriately calibrated in its confidence level?",
    ]
    
    def __init__(self, model, tokenizer, device="cuda"):
        ...
    
    def generate_with_critique(self, prompt: str) -> Dict[str, str]:
        """
        Generate response, critique it, and revise.
        
        Returns: {"initial": str, "critique": str, "revised": str}
        """
        ...
    
    def generate_training_pairs(self, prompts: List[str]) -> List[Dict]:
        """Generate (initial, revised) preference pairs for training."""
        ...
```

### PPO Trainer
Proximal Policy Optimization for RLHF. More powerful than DPO but requires a reward model.

```python
class RewardModel(nn.Module):
    """Reward model: scores responses given prompts."""
    
    def __init__(self, base_model: SelfImprovingLLM):
        super().__init__()
        self.encoder = base_model  # Or share architecture
        self.score_head = nn.Linear(base_model.config.d_model, 1)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return scalar reward for each sequence."""
        hidden = self.encoder(token_ids)["hidden_states"][:, -1, :]  # Last token
        return self.score_head(hidden).squeeze(-1)


class PPOTrainer:
    """
    PPO training for RLHF.
    
    Components:
    - Policy model (the LLM being trained)
    - Reference model (frozen, prevents divergence)
    - Reward model (scores completions)
    - Value model (estimates state values for advantage computation)
    
    Algorithm:
    1. Generate completions from policy
    2. Score completions with reward model
    3. Compute advantages using GAE
    4. Update policy with clipped objective
    5. Update value function
    """
    
    def __init__(
        self,
        policy_model: SelfImprovingLLM,
        ref_model: SelfImprovingLLM,
        reward_model: RewardModel,
        value_model: Optional[nn.Module] = None,
        clip_epsilon: float = 0.2,
        kl_coeff: float = 0.02,
        gamma: float = 0.99,
        lam: float = 0.95,
        learning_rate: float = 1e-5,
        device: str = "cuda",
    ):
        ...
    
    def train_step(self, prompts: List[str]) -> Dict[str, float]:
        """
        One PPO update step.
        
        Returns: {"policy_loss": float, "value_loss": float, "kl_div": float, "reward": float}
        """
        ...
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        log_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
        ...
```

---

## 5. Quantization (`selfllm/model/quantization.py`)

### GPTQ-style 4-bit Quantization

```python
class QuantizedLinear(nn.Module):
    """
    Linear layer with 4-bit weight quantization (GPTQ-style).
    
    Weights are stored in 4-bit integers, dequantized on-the-fly during forward pass.
    Uses grouped quantization: divide weights into groups, each with its own scale/zero_point.
    
    Memory reduction: ~4x (32-bit -> 4-bit)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        
        # Quantized weights stored as integers
        num_groups = (in_features * out_features) // group_size
        
        # Packed 4-bit weights (two weights per byte)
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8)
        )
        self.register_buffer('scales', torch.zeros(num_groups))
        self.register_buffer('zeros', torch.zeros(num_groups))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize weights and compute linear transformation."""
        # Dequantize
        weight = self.dequantize()
        return F.linear(x, weight)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize 4-bit weights to float32."""
        ...
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, bits: int = 4) -> "QuantizedLinear":
        """Quantize an existing Linear layer."""
        ...


def quantize_model(model: nn.Module, bits: int = 4) -> nn.Module:
    """Quantize all Linear layers in a model."""
    ...

def save_quantized(model: nn.Module, path: str) -> None:
    """Save quantized model (much smaller than full precision)."""
    ...

def load_quantized(path: str) -> nn.Module:
    """Load quantized model."""
    ...
```

### AWQ-style Activation-aware Quantization

```python
class AWQQuantizer:
    """
    AWQ: Activation-aware Weight Quantization.
    
    Key insight: Not all weights are equally important. Weights that correspond
    to large activation magnitudes should be quantized more carefully (or kept
    in higher precision).
    
    Algorithm:
    1. Run calibration data through model, collect activation magnitudes
    2. Scale weights channel-wise based on activation importance
    3. Quantize scaled weights
    4. Store scale factors alongside quantized weights
    """
    
    def calibrate(self, model, calibration_data):
        """Collect activation statistics on calibration data."""
        ...
    
    def quantize(self, model, bits=4):
        """Apply AWQ quantization."""
        ...
```
