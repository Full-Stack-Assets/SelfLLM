# Plan V4: Frontier Research Features

## Priority Ranking (by research impact + feasibility)

1. **Mixture of Experts (MoE)** — Sparse activation: scale to 1B+ effective params without proportional compute. Each token routes to top-k experts. GPT-4-class architecture.

2. **Multimodal (Vision + Text)** — ViT encoder projects image patches into LLM embedding space. Image understanding, captioning, VQA.

3. **Long Context Memory** — Sliding window attention + attention sinks + RAG vector retrieval. Break the 2048-token barrier.

4. **Constitutional AI + PPO** — Model critiques and revises its own outputs. True RLHF with PPO (beyond DPO).

5. **Quantization (GPTQ/AWQ)** — 4-bit/8-bit quantized serving. 4x memory reduction, essential for deploying MoE models.

---

## Architecture Impact Map

| Feature | Touches | New Modules |
|---------|---------|-------------|
| MoE | model/layers.py, model/model.py | model/moe.py |
| Multimodal | model/model.py, model/attention.py | model/vision.py |
| Long Context | model/attention.py, model/flash_attention.py | model/long_context.py |
| Constitutional AI + PPO | training/dpo_trainer.py, recursive/ | training/ppo_trainer.py, training/constitutional_ai.py |
| Quantization | model/model.py, serving/ | model/quantization.py |
