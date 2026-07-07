# AGENTS.md

## Cursor Cloud specific instructions

SelfLLM is a single pure-Python/PyTorch package (`selfllm`). There is **no database, cache, queue, or docker-compose** — nothing needs to be running just to develop or test. Everything runs CPU-only (CI runs on CPU; CUDA is optional and CUDA tests auto-skip).

The startup update script already installs all dependencies (CPU `torch==2.12.1+cpu`, `numpy==1.26.4`, `pip install -e ".[dev]"`, and `ruff==0.4.4`). Do not reinstall unless something is missing.

### Gotchas

- Only `python3` exists on this VM (there is no `python` alias), even though the README uses `python`. Use `python3 -m selfllm ...`.
- pip console scripts (`pytest`, `ruff`, `uvicorn`, `selfllm`, etc.) install to `~/.local/bin`, which is appended to `PATH` in `~/.bashrc`. If a fresh non-login shell can't find them, run `export PATH="$HOME/.local/bin:$PATH"` or invoke via module (`python3 -m pytest`, `python3 -m ruff`).
- `ruff` is required for lint but is intentionally NOT in `requirements.txt`/`setup.py` (CI installs it separately; the update script also installs it).

### Lint / Test / Build (project-wide, matches `.github/workflows/ci.yml`)

- Lint: `ruff check selfllm/ tests/ --select=E,F,W --ignore=E501,E402`. Note: `main` currently has many pre-existing lint findings (unused imports, etc.); this is not an environment problem.
- Tests: `pytest tests/ -q --cov=selfllm --cov-fail-under=78` (CI coverage gate is 78%; suite is ~752 passed, a few skipped on CPU). Slow tests need `--run-slow`; GPU tests auto-skip without CUDA.
- Build/package: `pip install -e .` (the package itself). Docs build (optional): `mkdocs build --strict` after `pip install -r requirements-docs.txt`.

### Running the product (OpenAI-compatible API server — the core service)

The serving entry point loads a model **directory** and a tokenizer **JSON file**:

```bash
python3 -m selfllm serve --model-path <model_dir> --tokenizer-path <tokenizer.json> --host 127.0.0.1 --port 8000
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `GET /v1/stats`. Set `SELFLLM_API_KEY` to require Bearer auth (off by default).

- Known caveat: `python3 -m selfllm init --save-path DIR` saves the model fine but then crashes with `IsADirectoryError` because it passes the directory to `tokenizer.save()`, which expects a **file path**. To get a usable tokenizer for serving, save it explicitly to a JSON file, e.g.:
  ```python
  from selfllm.model.tokenizer import BPETokenizer
  tok = BPETokenizer(vocab_size=1000); tok.train(["some corpus text"] * 16)
  tok.save("model_dir/tokenizer.json")
  ```
- A freshly-initialized (untrained) model will emit `<unk>`/garbage tokens — that is expected and still exercises the full tokenize → forward → generate → decode → OpenAI-JSON pipeline.

### CLI lifecycle commands

`python3 -m selfllm {init,pretrain,self-improve,real-train,generate,evaluate,serve,fsdp,ppo,benchmark}`. The README also mentions a `dashboard` command, but there is **no `dashboard` subparser** in the current CLI.
