# SPEC-v3.md — Production-Grade SelfLLM

## 1. Flash Attention 2 (`selfllm/model/flash_attention.py`)

Replace manual attention computation with Flash Attention 2 — a memory-efficient, IO-aware exact attention algorithm.

```python
class FlashAttention2(nn.Module):
    """Flash Attention 2 implementation with causal masking and KV cache support.
    
    Falls back to standard attention if flash_attn is not installed.
    Supports both prefill (full sequence) and decode (single token) modes.
    """
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float = 0.0):
        ...
    
    def forward(
        self,
        x: torch.Tensor,                    # [batch, seq_len, d_model]
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,  # For incremental decoding
        is_prefill: bool = True,            # True = full sequence, False = single token
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Returns:
            output: [batch, seq_len, d_model]
            kv_cache: Updated cache for incremental decoding (if provided)
        """
        ...

# Usage in attention.py: replace RoPEMultiHeadAttention.forward with FlashAttention2
# if flash_attn is available, else fall back to RoPEMultiHeadAttention

class HybridAttention(nn.Module):
    """Uses Flash Attention 2 if available, falls back to standard attention."""
    ...
```

**Key implementation**:
- Use `flash_attn.flash_attn_func` for the core computation
- Use `flash_attn.flash_attn_with_kvcache` for incremental decoding
- RoPE must be applied BEFORE Flash Attention (FA2 accepts pre-rotated Q, K)
- Causal masking via `causal=True` flag
- Automatic fallback if `flash_attn` package not installed
- Must handle both float16/bfloat16 (FA2 requirement) and float32 (testing on CPU)

## 2. Multi-GPU FSDP (`selfllm/training/fsdp_trainer.py`)

Fully Sharded Data Parallel for training models that don't fit on a single GPU.

```python
class FSDPTrainer:
    """
    FSDP training loop. Shards model parameters, gradients, and optimizer states
    across all GPUs. Each GPU holds only a fraction of the model.
    """
    
    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        config: Dict[str, Any],
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        device: str = "cuda",
    ):
        ...
    
    def setup_fsdp(self) -> None:
        """Wrap model with FSDP. Auto-wrap each TransformerBlock as a unit."""
        ...
    
    def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 10,
    ) -> Dict[str, List[float]]:
        """Distributed training loop with gradient sync across GPUs."""
        ...
    
    def save_checkpoint(self, path: str) -> None:
        """Save full model state (FSDP full state dict). Only rank 0 writes."""
        ...
    
    @staticmethod
    def setup_process_group() -> None:
        """Initialize torch.distributed process group."""
        ...
    
    @staticmethod
    def cleanup_process_group() -> None:
        """Destroy process group."""
        ...
```

**Key implementation**:
- `torch.distributed.fsdp.FullyShardedDataParallel`
- `auto_wrap_policy` wrapping each `TransformerBlock`
- `mixed_precision` policy (bf16 forward/backward, fp32 master weights)
- `backward_prefetch` for overlapping communication
- `CPUOffload` option for training models larger than GPU memory
- `FullStateDictType` for checkpoint saving
- Handle distributed sampler for data loading
- Only rank 0 logs, saves checkpoints, validates

## 3. PagedAttention + Serving

### 3.1 PagedAttention KV Cache (`selfllm/serving/paged_cache.py`)

```python
class BlockManager:
    """Manages KV cache as fixed-size blocks (like OS virtual memory pages).
    
    Each block holds KV for a fixed number of tokens (block_size, typically 16).
    Sequences share blocks via copy-on-write.
    """
    
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_heads: int, head_dim: int):
        ...
    
    def allocate(self, seq_len: int) -> List[int]:
        """Allocate blocks for a sequence. Returns block indices."""
        ...
    
    def append(self, block_ids: List[int]) -> List[int]:
        """Append a new block to a sequence. Handles block full -> allocate new."""
        ...
    
    def get_kv_cache(self, block_ids: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve KV tensors for given blocks. [num_blocks, block_size, num_heads, head_dim]"""
        ...
    
    def free(self, block_ids: List[int]) -> None:
        """Free blocks back to the pool."""
        ...


class PagedAttention(nn.Module):
    """Attention with PagedAttention KV cache backend."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, block_size: int = 16):
        ...
    
    def forward(
        self,
        x: torch.Tensor,
        block_manager: BlockManager,
        block_ids: List[int],
    ) -> torch.Tensor:
        """Compute attention using paged KV cache."""
        ...
```

### 3.2 Continuous Batching Scheduler (`selfllm/serving/scheduler.py`)

```python
class Request:
    """A single generation request."""
    id: str
    prompt_tokens: List[int]
    max_new_tokens: int
    temperature: float
    top_p: float
    status: str  # "waiting", "running", "finished"
    generated_tokens: List[int]
    block_ids: List[int]

class ContinuousBatchingScheduler:
    """
    vLLM-style continuous batching scheduler.
    
    - Maintains a queue of requests
    - Batches together all "running" requests every iteration
    - New requests are added to the running batch when space available
    - Finished requests are removed, their blocks freed
    """
    
    def __init__(self, model: SelfImprovingLLM, block_manager: BlockManager, max_batch_size: int = 32):
        ...
    
    def add_request(self, request: Request) -> None:
        ...
    
    def step(self) -> List[Request]:
        """
        Run one generation step for all active requests.
        Returns list of newly completed requests.
        """
        ...
    
    @property
    def num_active_requests(self) -> int: ...
```

### 3.3 OpenAI-Compatible API Server (`selfllm/serving/server.py`)

```python
"""
FastAPI server implementing OpenAI-compatible chat completions API.

Endpoints:
    POST /v1/chat/completions
    POST /v1/completions
    GET  /v1/models
    GET  /health
"""

from fastapi import FastAPI
from pydantic import BaseModel

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

app = FastAPI(title="SelfLLM API", version="1.0.0")

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    ...

@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completions endpoint."""
    ...

@app.get("/v1/models")
async def list_models():
    """List available models."""
    ...

@app.get("/health")
async def health():
    """Health check."""
    ...

def serve(
    model_path: str,
    tokenizer_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    max_batch_size: int = 32,
) -> None:
    """Launch the API server with PagedAttention backend."""
    ...
```

### 3.4 CLI Addition

Add `serve` subcommand to `train.py`:
```bash
python -m selfllm serve --model-path ./checkpoints/best --port 8000
```

## 4. Tool Use / Function Calling (`selfllm/tools/`)

### 4.1 Tool Registry & Execution (`selfllm/tools/registry.py`)

```python
@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    function: Callable          # Actual Python function

class ToolRegistry:
    """Registry of available tools."""
    
    def register(self, tool: Tool) -> None: ...
    def get(self, name: str) -> Tool: ...
    def list_tools(self) -> List[Dict[str, Any]]: ...
    def get_schemas(self) -> str: ...  # JSON schemas for prompt injection
    
    # Built-in tools:
    # - calculator: evaluate math expressions
    # - python: execute Python code in sandbox
    # - search: web search (placeholder)
    # - file_read: read file contents
    # - file_write: write file contents
    # - http_get: fetch URL content

class ToolExecutor:
    """Execute tool calls with error handling and timeouts."""
    
    def execute(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        tool_call: {"name": str, "arguments": Dict}
        Returns: {"status": "success|error", "result": Any, "error": Optional[str]}
        """
        ...
```

### 4.2 Function Calling Parser (`selfllm/tools/parser.py`)

```python
class FunctionCallParser:
    """Parse tool calls from model output."""
    
    # Supports multiple formats:
    FORMAT_XML = "xml"       # <tool>name</tool><args>{"x": 5}</args>
    FORMAT_JSON = "json"     # {"name": "calculator", "arguments": {"x": 5}}
    FORMAT_TAG = "tag"       # <function=calculator>{"x": 5}</function>
    
    def parse(self, text: str, format: str = "xml") -> Optional[Dict[str, Any]]:
        """Extract tool call from model output. Returns None if no tool call found."""
        ...
    
    def inject_tool_prompt(self, system_prompt: str, tools: List[Dict]) -> str:
        """Add tool definitions to system prompt so model knows available tools."""
        ...
```

### 4.3 Agent Loop (`selfllm/tools/agent.py`)

```python
class Agent:
    """
    ReAct-style agent: Reasoning + Acting loop.
    
    Loop:
        1. Model thinks about what to do (Reasoning)
        2. Model decides to use a tool or answer directly (Action)
        3. If tool: execute it, observe result (Observation)
        4. Feed observation back to model
        5. Repeat until answer or max iterations
    """
    
    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        tools: ToolRegistry,
        max_iterations: int = 10,
        device: str = "cuda",
    ):
        ...
    
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute agent loop on a query.
        
        Returns: {
            "answer": str,
            "tool_calls": List[Dict],
            "reasoning_trace": List[str],
            "iterations": int,
        }
        """
        ...
    
    def _format_history(self, interactions: List[Dict]) -> str:
        """Format interaction history for model context."""
        ...
```

## 5. Training Pipeline for Real Model (`selfllm/real_training.py`)

### 5.1 350M Parameter Config

```python
def get_350m_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=32000,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        d_ff=4096,
        max_seq_len=2048,
        dropout=0.1,
        use_rope=True,
        use_swiglu=True,
        grad_checkpoint=True,  # Enable gradient checkpointing for memory
    )

# ~350M parameters
```

### 5.2 End-to-End Training Script

```python
def train_real_model(
    num_books: int = 1000,
    pretrain_epochs: int = 5,
    self_improve_iterations: int = 10,
    use_fsdp: bool = True,
    use_flash_attn: bool = True,
    use_lora: bool = True,
    use_dpo: bool = True,
    output_dir: str = "./real_model",
):
    """
    Full pipeline:
    1. Download 1,000 Gutenberg books
    2. Train tokenizer on corpus
    3. Create training dataset
    4. Initialize 350M model with Flash Attention 2
    5. Pre-train with FSDP (multi-GPU)
    6. Run recursive self-improvement with LoRA + DPO
    7. Save final model
    """
    ...
```

### 5.3 CLI Command

```bash
python -m selfllm real-training \
    --num-books 1000 \
    --pretrain-epochs 5 \
    --self-improve-iterations 10 \
    --use-fsdp \
    --use-flash-attn \
    --use-lora \
    --use-dpo \
    --output-dir ./real_model
```
