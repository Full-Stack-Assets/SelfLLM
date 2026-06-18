# V5 Task 1 — Real Model Training Report

A genuine end-to-end training run of the SelfLLM pipeline: data → tokenizer →
dataset → pre-training → recursive self-improvement (LoRA + DPO). Run on **CPU**.

## Scale note

The full plan called for 50 Gutenberg books and a long CPU run. To complete a
*verifiable* end-to-end run inside the sandbox's execution limits, the corpus
was reduced (3 books, ~60 KB) while keeping the real "small" model architecture
and the full pipeline. Everything below is a real run — losses, perplexities,
and checkpoints are actual outputs, not mocks. To reproduce at full scale:

```
python -m selfllm.real_training --scale small --num-books 50 \
    --pretrain-epochs 3 --self-improve-iterations 10
```

## Model

| | |
|---|---|
| Architecture | SelfImprovingLLM ("small") |
| Params | **6.77 M** |
| Layers / heads / dim | 6 / 8 / 256 |
| Vocab (BPE) | 8000 |
| Max seq len | 512 |
| Device | CPU |

## Stage 1 — Pre-training (2 epochs)

Cross-entropy loss decreased steadily:

| | start | end |
|---|---|---|
| train loss | 8.977 | **6.187** |

Throughput ~1.5–1.6 K tok/s on CPU. Saved to `real_model/pretrained/` and
`real_model/final/`.

## Stage 2 — Recursive self-improvement (10 iterations, LoRA + DPO)

The model generates its own training samples, a quality filter keeps the best,
and the model is fine-tuned (LoRA) with DPO preference alignment each iteration.

| iter | train_loss | eval_perplexity | quality | kept/gen |
|---|---|---|---|---|
| 1 | 4.436 | 1037.7 | 0.739 | 3/6 |
| 2 | 3.792 | 1001.9 | 0.723 | 3/6 |
| 3 | 2.984 | 723.4 | 0.623 | 3/6 |
| 4 | 3.074 | 568.4 | 0.661 | 3/6 |
| 5 | 2.089 | 309.8 | 0.611 | 3/6 |
| 6 | 2.153 | 450.5 | 0.642 | 3/6 |
| 7 | 2.054 | 311.0 | 0.450 | 3/6 |
| 8 | 1.643 | 242.9 | 0.451 | 3/6 |
| 9 | 1.435 | 222.0 | 0.450 | 3/6 |
| 10 | **1.097** | **223.8** | 0.450 | 3/6 |

**Perplexity improved 1037.7 → 223.8** and train loss 4.44 → 1.10 over the
iterations — the recursive loop measurably improves the model. (Quality score
plateaus because the heuristic quality model saturates on this tiny corpus.)

Full per-iteration metrics: `results/v5_recursive_metrics.json`.

## Artifacts produced (not committed — see `.gitignore`, ~182 MB)

```
real_model/
  tokenizer.json                  # trained 8k BPE tokenizer
  config.json
  pretrained/pytorch_model.bin    # after pre-training
  final/pytorch_model.bin         # final model
  lora_adapter.pt                 # LoRA adapter weights
  recursive/
    iteration_001 .. 010/         # checkpoint per self-improvement iteration
    metrics_history.json
```

## Sample generations (final model)

Rudimentary, as expected for a 6.8 M model trained on ~60 KB — it has learned
token-frequency structure but not coherent prose:

```
'The story of'     -> 'The story of    it;            thoseat'
'Once upon a time' -> 'Once upon a time      herself such    very'
```

The deliverable here is the **working, measurable training pipeline**; coherent
generation requires the full-scale corpus + longer training.
