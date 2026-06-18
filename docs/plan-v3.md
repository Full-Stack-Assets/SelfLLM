# Plan V3: Production-Grade SelfLLM

## Priority: 5 Major Enhancements

1. **Flash Attention 2** — Replace manual attention matmuls with FA2 for 2-4x speedup + 4x memory reduction
2. **Multi-GPU FSDP** — Fully Sharded Data Parallel for training 350M+ parameter models
3. **PagedAttention + Serving** — vLLM-style KV cache paging, continuous batching, OpenAI-compatible API
4. **Tool Use / Function Calling** — Structured tool calls, execution engine, agent loop
5. **Train a Real Model** — 350M parameter model trained on 1,000+ Gutenberg books with full recursive self-improvement

---

## Execution Strategy

### Wave 1: Core Infrastructure (Flash Attention 2 + FSDP + PagedAttention)
- These all touch the model's forward pass and memory layout
- Flash Attention 2 replaces manual attention computation
- FSDP wraps the model for multi-GPU sharding
- PagedAttention replaces KV cache storage with non-contiguous blocks

### Wave 2: Application Layer (Tool Use + Function Calling + Agent Loop)
- Tool definitions, execution engine, structured output parsing
- Agent loop: plan → execute tools → observe → critique → repeat

### Wave 3: Serving (OpenAI-compatible API Server)
- FastAPI server with /v1/chat/completions
- PagedAttention integration for batched serving
- Continuous batching of incoming requests

### Wave 4: Training Pipeline for Real Model
- 350M parameter config
- Download 1,000+ Gutenberg books
- Full pre-training + recursive self-improvement with LoRA + DPO
- Multi-GPU FSDP training

---

## Skill: vibecoding-general-swarm (Mode A — multi-agent)
