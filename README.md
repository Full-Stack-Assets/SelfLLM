# SelfLLM: A Recursively Self-Improving Foundation Language Model

> **Key Innovation**: A closed-loop system where the LLM is its own teacher, student, and critic -- recursively generating, filtering, and learning from self-produced data.

SelfLLM is a complete, from-scratch implementation of a foundation language model with a recursive self-training pipeline. The model generates its own training data, filters by quality, and re-trains iteratively to improve itself without external human-labeled data.

**v2.0 Enhancement Suite**: LoRA fine-tuning, DPO alignment, Chain-of-Thought reasoning, speculative decoding, real data pipeline, and a Gradio web dashboard.

---

## Architecture Overview

### Recursive Self-Improvement Loop

```
+---------------------------------------------------------------------+
|                    RECURSIVE SELF-IMPROVEMENT LOOP                  |
|                                                                     |
|  +----------+    +--------------+    +--------------+              |
|  |  Model   |--->|   Generate   |--->|    Filter    |              |
|  | (v_n)    |    |   Responses  |    |   (Quality)  |              |
|  +----------+    +--------------+    +------+-------+              |
|       ^                                     |                       |
|       |                                     v                       |
|  +----+----+    +--------------+    +--------------+              |
|  |  Model  |<---|    Train     |<---|    Curate    |              |
|  | (v_n+1) |    |  (Self-Train)|    |   Dataset    |              |
|  +---------+    +--------------+    +--------------+              |
+---------------------------------------------------------------------+
```

### Enhancement Suite (v2.0)

```
+---------------------------------------------------------------+
|                      ENHANCEMENT LAYERS                        |
+---------------------------------------------------------------+
|                                                                |
|  +----------+  +----------+  +-----------+  +-------------+  |
|  |   LoRA   |  |   DPO    |  |    CoT    |  |  SpecDec   |  |
|  | Fine-Tune|  | Align    |  | Reasoning |  | Fast Gen   |  |
|  |(train    |  |(preference|  |(<think>   |  |(draft     |  |
|  | adapters)|  | learning) |  | /answer>) |  | model)     |  |
|  +----------+  +----------+  +-----------+  +-------------+  |
|                                                                |
|  +----------+  +------------------------------------------+  |
|  |  Data    |  |              Web Dashboard                |  |
|  | Pipeline |  |  (Generate / Self-Improve / Train / Eval) |  |
|  | (Gutenberg|  +------------------------------------------+  |
|  |  books)  |                                                  |
|  +----------+                                                  |
+---------------------------------------------------------------+
```

---

## Components

| Module | Description | Key Features |
|--------|-------------|--------------|
| `selfllm/model/` | Core transformer model | RoPE, SwiGLU, causal attention, KV cache, quality scoring |
| `selfllm/model/lora.py` | LoRA fine-tuning | Rank-4/8 adapters, inject/merge/save/load, <1% trainable params |
| `selfllm/model/speculative.py` | Speculative decoding | Small draft model, 2-3x faster generation |
| `selfllm/cot/` | Chain-of-Thought reasoning | `<think>`/`<answer>` tags, self-consistency voting |
| `selfllm/data/` | Real data pipeline | Gutenberg downloader, preprocessing, chunking |
| `selfllm/training/` | Training pipeline | Data generator, quality filter, DPO trainer, standard trainer |
| `selfllm/training/dpo_trainer.py` | DPO alignment | Preference learning without reward model |
| `selfllm/recursive/` | Recursive engine | Iteration loop, evaluator, experience replay, rollback |
| `selfllm/dashboard.py` | Gradio Web UI | Generate, self-improve, train, evaluate, settings tabs |
| `selfllm/train.py` | Full CLI | 5 subcommands for entire lifecycle |

---

## Quick Start

### Installation

```bash
git clone <repo>
cd selfllm
pip install -e .
```

### 1. Initialize Model & Tokenizer

```bash
python -m selfllm init \
    --vocab-size 32000 \
    --d-model 512 \
    --n-layers 8 \
    --n-heads 8 \
    --save-path ./my_model
```

### 2. Download Training Data

```python
from selfllm.data.pipeline import DataPipeline
from selfllm.model.tokenizer import BPETokenizer

tokenizer = BPETokenizer(vocab_size=32000)
pipeline = DataPipeline(tokenizer)

# Download 50 public domain books from Project Gutenberg
pipeline.download_gutenberg_books("./data/books", num_books=50)

# Create training dataset
dataset = pipeline.create_training_dataset("./data/books", output_path="./data/train.pt")
```

### 3. Pre-train

```bash
python -m selfllm pretrain \
    --data-path ./data/train.pt \
    --num-epochs 10 \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --checkpoint-dir ./checkpoints
```

### 4. Launch Web Dashboard

```bash
python -m selfllm dashboard --port 7860
# Or programmatically:
from selfllm.dashboard import launch_dashboard
launch_dashboard(model, tokenizer, port=7860)
```

### 5. Run Recursive Self-Improvement

```bash
python -m selfllm self-improve \
    --model-path ./checkpoints/best \
    --max-iterations 10 \
    --samples-per-iteration 2000 \
    --use-lora --lora-rank 8 \
    --use-dpo --dpo-beta 0.1
```

### 6. Generate with Chain-of-Thought

```python
from selfllm.cot.cot_generator import ChainOfThoughtGenerator

cot = ChainOfThoughtGenerator(model, tokenizer)
result = cot.generate_cot_response("What is 15 * 23?")
print(result["thinking"])  # Step-by-step reasoning
print(result["answer"])    # Final answer

# Self-consistency voting (5 reasoning paths, pick majority)
vote = cot.self_consistency_vote("What is 15 * 23?", num_paths=5)
print(f"Answer: {vote['answer']} (confidence: {vote['confidence']*100:.0f}%)")
```

### 7. Speed Up Generation with Speculative Decoding

```python
from selfllm.model.speculative import SpeculativeDecoder

spec = SpeculativeDecoder(model, gamma=4)
output = spec.generate(prompt_ids, max_new_tokens=100)
print(f"Accepted {spec.acceptance_rate*100:.1f}% of draft tokens")
```

---

## Key Features Explained

### LoRA (Low-Rank Adaptation)
Instead of fine-tuning all model parameters during recursive self-improvement, LoRA injects small rank-r adapter matrices into attention and FFN layers. Only these adapters are trained -- the base model stays frozen.

- **Efficiency**: 99%+ of parameters frozen, training is 10x faster
- **Checkpoints**: LoRA weights are tiny (~22KB vs hundreds of MB)
- **Merging**: After training, adapters can be merged into base weights for fast inference

### DPO (Direct Preference Optimization)
Replaces the simple quality filter with proper preference learning. Given pairs of (chosen, rejected) responses, DPO directly optimizes the policy:

```
L_DPO = -log sigmoid(beta * (log_pi(chosen) - log_pi(rejected)))
```

- **No reward model needed** -- simpler than PPO/RLHF
- **Self-critic**: Generates N responses, picks best/worst as preference pair
- **Beta parameter**: Controls deviation from reference model (0.1 = conservative)

### Chain-of-Thought Reasoning
Trains the model to generate step-by-step reasoning before answering:

```
<think>First, I'll multiply 12 by 20 to get 240.
Then, multiply 12 by 3 to get 36.
Finally, add 240 + 36 = 276.</think>
<answer>276</answer>
```

- **Self-consistency voting**: Generate 5 reasoning paths, pick the most common answer
- **Math & logic problems**: Built-in templates for arithmetic, word problems, logic puzzles

### Speculative Decoding
Uses a small draft model (4x fewer parameters) to generate candidate tokens, which the target model verifies in a single parallel forward pass:

- **2-3x faster** generation on average
- **Adaptive acceptance**: Tokens accepted if `p_target / p_draft > uniform(0,1)`
- **Auto-created draft**: Smaller model derived automatically from target config

### Experience Replay
Prevents catastrophic forgetting by mixing new self-generated data with past high-quality samples (30% replay ratio by default).

### Automatic Rollback
If an iteration degrades performance, the system automatically reverts to the previous best checkpoint.

---

## CLI Reference

| Command | Description | Key Flags |
|---------|-------------|-----------|
| `selfllm init` | Create model + tokenizer | `--vocab-size`, `--d-model`, `--n-layers` |
| `selfllm pretrain` | Pre-train on corpus | `--data-path`, `--num-epochs`, `--batch-size` |
| `selfllm self-improve` | Recursive improvement loop | `--max-iterations`, `--use-lora`, `--use-dpo` |
| `selfllm generate` | Interactive text generation | `--prompt`, `--temperature`, `--top-p` |
| `selfllm evaluate` | Run evaluation suite | `--eval-data`, `--output-path` |
| `selfllm dashboard` | Launch web UI | `--port`, `--share` |

---

## Project Structure

```
selfllm/
├── __init__.py
├── __main__.py              # CLI entry point
├── config.yaml              # Default configuration
├── train.py                 # CLI implementation (5 subcommands)
├── utils.py                 # Utilities: seeding, device, logging
├── dashboard.py             # Gradio Web UI
├── model/
│   ├── __init__.py
│   ├── config.py            # ModelConfig dataclass
│   ├── tokenizer.py         # BPE Tokenizer
│   ├── attention.py         # RoPE Multi-Head Attention
│   ├── layers.py            # TransformerBlock, SwiGLU, RMSNorm
│   ├── model.py             # SelfImprovingLLM (main model)
│   ├── quality_model.py     # Quality scoring head
│   ├── lora.py              # LoRA adapters (NEW v2)
│   └── speculative.py       # Speculative decoding (NEW v2)
├── training/
│   ├── __init__.py
│   ├── data_generator.py    # Self-training data generation
│   ├── quality_filter.py    # Multi-dimensional quality filtering
│   ├── dataset.py           # PyTorch Dataset
│   ├── trainer.py           # Standard trainer
│   └── dpo_trainer.py       # DPO alignment trainer (NEW v2)
├── recursive/
│   ├── __init__.py
│   ├── recursive_config.py  # RecursiveConfig dataclass
│   ├── recursive_trainer.py # Main recursive loop
│   └── evaluator.py         # Evaluation suite
├── cot/                     # Chain-of-Thought (NEW v2)
│   ├── __init__.py
│   └── cot_generator.py
└── data/                    # Real data pipeline (NEW v2)
    ├── __init__.py
    └── pipeline.py

tests/                       # 116 tests across 8 test files
├── test_tokenizer.py        # 8 tests
├── test_model.py            # 9 tests
├── test_generation.py       # 6 tests
├── test_recursive.py        # 7 tests
├── test_lora.py             # 32 tests (NEW v2)
├── test_dpo.py              # 17 tests (NEW v2)
├── test_cot.py              # 16 tests (NEW v2)
└── test_speculative.py      # 20 tests (NEW v2)

setup.py
requirements.txt
README.md
```

---

## Configuration

Edit `selfllm/config.yaml` or pass `--config` to any command:

```yaml
model:
  vocab_size: 32000
  d_model: 512
  n_layers: 8
  n_heads: 8
  d_ff: 2048
  max_seq_len: 512

training:
  batch_size: 16
  learning_rate: 5e-4
  num_epochs: 10

recursive:
  use_lora: true              # Enable LoRA (NEW)
  lora_rank: 8                # LoRA rank
  lora_alpha: 16.0            # LoRA scaling
  use_dpo: true               # Enable DPO (NEW)
  dpo_beta: 0.1               # DPO temperature
  samples_per_iteration: 2000
  keep_ratio: 0.4
  max_iterations: 10
```

---

## Stats

| Metric | Value |
|--------|-------|
| Total Python LOC | 10,103 |
| Number of files | 38 |
| Test coverage | 116 tests, all passing |
| Packages | 7 (model, training, recursive, cot, data, utils, dashboard) |
| CLI commands | 6 |
| Web UI tabs | 5 |

---

## License

MIT License - See LICENSE file for details.
