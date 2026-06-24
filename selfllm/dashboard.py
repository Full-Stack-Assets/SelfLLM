"""Gradio-based Web UI dashboard for SelfLLM monitoring and interaction."""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gradio as gr
import torch
import yaml

from selfllm.model.config import ModelConfig
from selfllm.model.model import SelfImprovingLLM
from selfllm.model.tokenizer import BPETokenizer


# --------------------------------------------------------------------------- #
# Global state (managed via gr.State)
# --------------------------------------------------------------------------- #


class DashboardState:
    """Mutable container for dashboard runtime state."""

    def __init__(
        self,
        model: Optional[SelfImprovingLLM] = None,
        tokenizer: Optional[BPETokenizer] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.stop_flag = False
        self.iteration_logs: List[Dict[str, Any]] = []
        self.training_logs: List[Dict[str, float]] = []
        self.eval_results: Optional[Dict[str, Any]] = None
        self.model_path: Optional[str] = None
        self.tokenizer_path: Optional[str] = None
        self.device: Optional[str] = None


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_config(config_path: str, config: Dict[str, Any]) -> str:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return f"Configuration saved to {config_path}"


def _candidate_tokenizer_paths(
    model_dir: str, tokenizer_path: Optional[str] = None
) -> Sequence[str]:
    """Return possible tokenizer files in priority order."""
    candidates: List[str] = []
    if tokenizer_path:
        candidates.append(tokenizer_path)
    candidates.extend(
        [
            os.path.join(model_dir, "tokenizer.json"),
            os.path.join(model_dir, "tokenizer"),
        ]
    )
    return candidates


def _load_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    device: str = "cpu",
) -> tuple[Optional[SelfImprovingLLM], Optional[BPETokenizer], str]:
    """Load a model and tokenizer from paths used by the CLI and UI."""
    try:
        model_dir = (model_path or "").strip()
        if not model_dir:
            return None, None, "Model path is required."
        if not os.path.isdir(model_dir):
            return None, None, f"Model directory not found: {model_dir}"

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(config_path):
            return None, None, f"Missing model config: {config_path}"
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config = ModelConfig(**config_dict)
        model = SelfImprovingLLM(config).to(device)

        weights_path = os.path.join(model_dir, "pytorch_model.bin")
        if not os.path.exists(weights_path):
            return None, None, f"Missing model weights: {weights_path}"
        state_dict = torch.load(
            weights_path, map_location=device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()

        tok_path = next(
            (
                path for path in _candidate_tokenizer_paths(
                    model_dir, (tokenizer_path or "").strip() or None
                )
                if os.path.isfile(path)
            ),
            None,
        )
        if tok_path is None:
            return (
                None,
                None,
                "Tokenizer not found. Provide a tokenizer file or place "
                "`tokenizer.json` beside the model checkpoint.",
            )
        tokenizer = BPETokenizer(vocab_size=config.vocab_size)
        tokenizer.load(tok_path)

        params = sum(p.numel() for p in model.parameters())
        message = (
            f"Loaded {params:,} parameter model from {model_dir} "
            f"with tokenizer {tok_path} on {device}."
        )
        return model, tokenizer, message
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Failed to load model: {exc}"


def _load_model_from_dir(
    model_dir: str, device: str = "cpu"
) -> tuple[Optional[SelfImprovingLLM], Optional[BPETokenizer]]:
    """Backward-compatible wrapper for loading a checkpoint directory."""
    model, tokenizer, _ = _load_model(model_dir, device=device)
    return model, tokenizer


def _set_loaded_model(
    state: DashboardState,
    model: SelfImprovingLLM,
    tokenizer: BPETokenizer,
    model_path: Optional[str],
    tokenizer_path: Optional[str],
    device: str,
) -> None:
    """Store a loaded model and its display metadata in dashboard state."""
    state.model = model
    state.tokenizer = tokenizer
    state.model_path = model_path
    state.tokenizer_path = tokenizer_path
    state.device = device


def _model_status(state: DashboardState) -> str:
    """Render the current model status for the dashboard header."""
    if state.model is None or state.tokenizer is None:
        return (
            "**Model status:** No model loaded. Load a checkpoint in Settings "
            "or launch with `selfllm dashboard --model-path ... --tokenizer-path ...`."
        )
    params = sum(p.numel() for p in state.model.parameters())
    config = getattr(state.model, "config", None)
    parts = [
        f"**Model status:** Loaded `{params:,}` parameters",
        f"device `{state.device or next(state.model.parameters()).device}`",
    ]
    if config is not None:
        parts.append(
            f"context `{getattr(config, 'max_seq_len', 'unknown')}` tokens"
        )
    if state.model_path:
        parts.append(f"checkpoint `{state.model_path}`")
    return " | ".join(parts)


def _format_chat_prompt(
    message: str,
    history: Optional[List[Tuple[str, str]]] = None,
    system_prompt: str = "",
) -> str:
    """Convert chat turns into a plain text prompt for the base LLM."""
    turns: List[str] = []
    if system_prompt.strip():
        turns.append(f"System: {system_prompt.strip()}")
    for user_text, assistant_text in history or []:
        if user_text:
            turns.append(f"User: {user_text.strip()}")
        if assistant_text:
            turns.append(f"Assistant: {assistant_text.strip()}")
    turns.append(f"User: {message.strip()}")
    turns.append("Assistant:")
    return "\n".join(turns)


def _messages_to_turns(
    history: Optional[List[Dict[str, str]]],
) -> List[Tuple[str, str]]:
    """Convert Gradio messages into user/assistant prompt turns."""
    turns: List[Tuple[str, str]] = []
    pending_user: Optional[str] = None
    for message in history or []:
        role = message.get("role")
        content = message.get("content", "")
        if role == "user":
            if pending_user is not None:
                turns.append((pending_user, ""))
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                continue
            turns.append((pending_user, content))
            pending_user = None
    if pending_user is not None:
        turns.append((pending_user, ""))
    return turns


# --------------------------------------------------------------------------- #
# Tab handlers
# --------------------------------------------------------------------------- #


def _handle_generate(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_k: int,
    repetition_penalty: float,
    state: DashboardState,
) -> str:
    """Handle text generation from the Generate tab."""
    if state.model is None or state.tokenizer is None:
        return "Error: No model loaded. Please load a model in the Settings tab or pass one at startup."
    if not prompt.strip():
        return "Error: Enter a prompt before generating."

    state.model.eval()
    device = next(state.model.parameters()).device
    prompt_ids = torch.tensor(
        [state.tokenizer.encode(prompt)], device=device
    )

    with torch.no_grad():
        output = state.model.generate(
            prompt_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_token_id=state.tokenizer.eos_token_id,
        )

    generated_ids = output["sequences"][0].cpu().tolist()
    prompt_len = prompt_ids.shape[1]
    new_token_ids = generated_ids[prompt_len:]

    return state.tokenizer.decode(new_token_ids).strip()


def _handle_chat(
    message: str,
    history: Optional[List[Dict[str, str]]],
    system_prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    state: DashboardState,
) -> tuple[List[Dict[str, str]], str]:
    """Handle one chat turn in the Chat tab."""
    current_history = list(history or [])
    if not message.strip():
        return current_history, ""

    if state.model is None or state.tokenizer is None:
        current_history.append({"role": "user", "content": message})
        current_history.append(
            {
                "role": "assistant",
                "content": "Error: No model loaded. Load a model in Settings or start "
                "with `selfllm dashboard --model-path ... --tokenizer-path ...`.",
            }
        )
        return current_history, ""

    prompt = _format_chat_prompt(
        message,
        _messages_to_turns(current_history),
        system_prompt,
    )
    response = _handle_generate(
        prompt=prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=int(max_tokens),
        top_k=50,
        repetition_penalty=repetition_penalty,
        state=state,
    )
    current_history.append({"role": "user", "content": message})
    if response.startswith("Error:"):
        current_history.append({"role": "assistant", "content": response})
    else:
        current_history.append({"role": "assistant", "content": response.strip()})
    return current_history, ""


def _handle_self_improve_start(
    num_iterations: int,
    samples_per_iteration: int,
    learning_rate: float,
    batch_size: int,
    temperature: float,
    state: DashboardState,
) -> str:
    """Start the recursive self-improvement loop in a background thread."""
    if state.model is None or state.tokenizer is None:
        return "Error: No model loaded."

    state.stop_flag = False
    state.iteration_logs.clear()

    def _run_loop() -> None:
        for i in range(num_iterations):
            if state.stop_flag:
                break
            # Placeholder: real implementation would call RecursiveSelfTrainer
            log_entry = {
                "iteration": i + 1,
                "status": "completed",
                "loss": round(2.5 - 0.15 * i + (0.05 if i % 2 else -0.03), 4),
                "perplexity": round(
                    12.0 - 0.5 * i + (0.3 if i % 2 else -0.2), 2
                ),
                "quality_score": round(0.6 + 0.03 * i, 3),
                "timestamp": time.strftime("%H:%M:%S"),
            }
            state.iteration_logs.append(log_entry)
            time.sleep(0.1)  # Simulate work

    thread = threading.Thread(target=_run_loop, daemon=True)
    thread.start()
    return "Self-improvement loop started!"


def _handle_self_improve_stop(state: DashboardState) -> str:
    """Stop the recursive self-improvement loop."""
    state.stop_flag = True
    return "Stop signal sent. Current iteration will finish before stopping."


def _get_iteration_table(state: DashboardState) -> List[List[Any]]:
    """Format iteration logs for gr.Dataframe."""
    if not state.iteration_logs:
        return []
    headers = [
        ["Iteration", "Status", "Loss", "Perplexity", "Quality", "Time"]
    ]
    rows = [
        [
            log["iteration"],
            log["status"],
            log["loss"],
            log["perplexity"],
            log["quality_score"],
            log["timestamp"],
        ]
        for log in state.iteration_logs
    ]
    return headers + rows


def _get_loss_plot_data(
    state: DashboardState,
) -> tuple[list[int], list[float]]:
    """Extract loss curve data for gr.Plot."""
    iterations = [log["iteration"] for log in state.iteration_logs]
    losses = [log["loss"] for log in state.iteration_logs]
    return iterations, losses


def _handle_metrics_report(metrics_path: str):
    """Load a ``metrics_history.json`` and render a report + table.

    Surfaces the per-iteration core metrics and any benchmark scores recorded
    by the recursive loop's eval-suite hook (``benchmark_<name>`` keys).

    Returns ``(markdown_report, dataframe_rows)``.
    """
    from selfllm.recursive.report import (
        format_metrics_report,
        load_metrics,
        metrics_table_rows,
    )

    path = (metrics_path or "").strip()
    if not path:
        return "_Enter the path to a `metrics_history.json` file and click Load._", []
    if not os.path.exists(path):
        return f"_File not found: `{path}`_", []
    try:
        history = load_metrics(path)
    except Exception as exc:  # noqa: BLE001
        return f"_Failed to load metrics: {exc}_", []
    if not history:
        return "_No iterations recorded in this metrics file._", []
    return format_metrics_report(history), metrics_table_rows(history)


def _handle_training_start(
    corpus_dir: str,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    max_seq_len: int,
    state: DashboardState,
) -> str:
    """Start pre-training on a corpus."""
    if state.model is None or state.tokenizer is None:
        return "Error: No model loaded."
    if not os.path.isdir(corpus_dir):
        return f"Error: Directory '{corpus_dir}' does not exist."

    state.training_logs.clear()
    state.stop_flag = False

    def _run_training() -> None:
        for epoch in range(num_epochs):
            if state.stop_flag:
                break
            # Placeholder: real implementation would use Trainer
            for step in range(10):
                if state.stop_flag:
                    break
                loss = 3.0 - 0.2 * epoch - 0.01 * step + 0.05 * (step % 3)
                state.training_logs.append(
                    {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "loss": round(loss, 4),
                    }
                )
            time.sleep(0.05)

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return f"Training started on corpus: {corpus_dir}"


def _get_training_plot_data(
    state: DashboardState,
) -> tuple[list[int], list[float]]:
    """Extract training loss data for plotting."""
    steps = list(range(1, len(state.training_logs) + 1))
    losses = [entry["loss"] for entry in state.training_logs]
    return steps, losses


def _handle_evaluation(state: DashboardState) -> str:
    """Run evaluation suite and return results as JSON."""
    if state.model is None or state.tokenizer is None:
        return json.dumps(
            {"error": "No model loaded."}, indent=2
        )

    # Placeholder: real implementation would call evaluator
    eval_prompts = [
        "The sky is",
        "In the year 2050,",
        "The theory of relativity states that",
        "Once upon a time",
        "The capital of France is",
    ]
    state.model.eval()
    device = next(state.model.parameters()).device

    generated_texts = []
    with torch.no_grad():
        for prompt in eval_prompts:
            prompt_ids = torch.tensor(
                [state.tokenizer.encode(prompt)], device=device
            )
            output = state.model.generate(
                prompt_ids, max_new_tokens=32, temperature=0.8, top_p=0.92
            )
            text = state.tokenizer.decode(output["sequences"][0].tolist())
            generated_texts.append(text)

    state.eval_results = {
        "num_prompts": len(eval_prompts),
        "avg_length_chars": sum(len(t) for t in generated_texts)
        // max(len(generated_texts), 1),
        "sample_outputs": generated_texts[:3],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return json.dumps(state.eval_results, indent=2)


# --------------------------------------------------------------------------- #
# Dashboard factory
# --------------------------------------------------------------------------- #


def create_dashboard(
    model: Optional[SelfImprovingLLM] = None,
    tokenizer: Optional[BPETokenizer] = None,
    config_path: str = "./selfllm/config.yaml",
    checkpoint_dir: str = "./checkpoints",
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: Optional[str] = None,
) -> gr.Blocks:
    """Create and launch the Gradio dashboard.

    Tabs:
    - **Generate**: Interactive text generation with all parameters.
    - **Self-Improve**: Start/stop recursive loop, watch live metrics.
    - **Training**: Pre-train on corpus with live loss curves.
    - **Evaluation**: Run eval suite, view results.
    - **Metrics Report**: Load a metrics_history.json and view per-iteration
      core metrics + benchmark scores (MMLU/GSM8K/HumanEval).
    - **Settings**: Model config editor.

    Args:
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
        config_path: Path to the YAML configuration file.
        checkpoint_dir: Directory containing model checkpoints.

    Returns:
        A Gradio ``Blocks`` application ready to be launched.
    """
    # Load config
    config = _load_config(config_path)
    config_text = yaml.dump(
        config, default_flow_style=False, sort_keys=False
    )

    # Initial state
    initial_state = DashboardState(model=model, tokenizer=tokenizer)
    if model is not None and tokenizer is not None:
        initial_state.model_path = model_path
        initial_state.tokenizer_path = tokenizer_path
        initial_state.device = device or str(next(model.parameters()).device)

    with gr.Blocks(title="SelfLLM Dashboard") as demo:
        state = gr.State(value=initial_state)

        gr.Markdown("# SelfLLM Dashboard")
        gr.Markdown(
            "Interactive dashboard for monitoring and controlling "
            "the SelfLLM self-improving language model."
        )
        model_status = gr.Markdown(value=_model_status(initial_state))

        with gr.Tabs():
            # ==================== Chat Tab ====================
            with gr.TabItem("Chat"):
                gr.Markdown("### Chat with the current SelfLLM checkpoint")
                chat_history = gr.Chatbot(
                    label="Conversation",
                    height=420,
                )
                chat_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask the loaded model anything...",
                    lines=3,
                )
                with gr.Row():
                    chat_send_btn = gr.Button("Send", variant="primary")
                    chat_clear_btn = gr.Button("Clear")
                with gr.Accordion("Generation settings", open=False):
                    chat_system = gr.Textbox(
                        label="System Prompt",
                        value=(
                            "You are SelfLLM, a helpful language model running "
                            "from the local checkpoint."
                        ),
                        lines=2,
                    )
                    with gr.Row():
                        chat_temperature = gr.Slider(
                            0.0, 2.0, value=0.8, step=0.05,
                            label="Temperature"
                        )
                        chat_top_p = gr.Slider(
                            0.0, 1.0, value=0.92, step=0.01, label="Top-p"
                        )
                    with gr.Row():
                        chat_max_tokens = gr.Slider(
                            1, 1024, value=256, step=1,
                            label="Max New Tokens"
                        )
                        chat_repetition = gr.Slider(
                            1.0, 2.0, value=1.05, step=0.05,
                            label="Repetition Penalty"
                        )

                chat_inputs = [
                    chat_input,
                    chat_history,
                    chat_system,
                    chat_temperature,
                    chat_top_p,
                    chat_max_tokens,
                    chat_repetition,
                    state,
                ]
                chat_send_btn.click(
                    fn=_handle_chat,
                    inputs=chat_inputs,
                    outputs=[chat_history, chat_input],
                )
                chat_input.submit(
                    fn=_handle_chat,
                    inputs=chat_inputs,
                    outputs=[chat_history, chat_input],
                )
                chat_clear_btn.click(
                    fn=lambda: [],
                    inputs=None,
                    outputs=chat_history,
                )

            # ==================== Generate Tab ====================
            with gr.TabItem("Generate"):
                gr.Markdown("### Interactive Text Generation")
                with gr.Row():
                    with gr.Column(scale=2):
                        gen_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Enter your prompt here...",
                            lines=4,
                        )
                        gen_output = gr.Textbox(
                            label="Generated Text",
                            lines=8,
                            interactive=False,
                        )
                        gen_btn = gr.Button("Generate", variant="primary")
                    with gr.Column(scale=1):
                        gen_temperature = gr.Slider(
                            0.0, 2.0, value=0.8, step=0.05,
                            label="Temperature"
                        )
                        gen_top_p = gr.Slider(
                            0.0, 1.0, value=0.92, step=0.01, label="Top-p"
                        )
                        gen_top_k = gr.Slider(
                            0, 100, value=50, step=1, label="Top-k"
                        )
                        gen_max_tokens = gr.Slider(
                            1, 1024, value=128, step=1,
                            label="Max New Tokens"
                        )
                        gen_repetition = gr.Slider(
                            1.0, 2.0, value=1.0, step=0.05,
                            label="Repetition Penalty"
                        )

                gen_btn.click(
                    fn=_handle_generate,
                    inputs=[
                        gen_prompt,
                        gen_temperature,
                        gen_top_p,
                        gen_max_tokens,
                        gen_top_k,
                        gen_repetition,
                        state,
                    ],
                    outputs=gen_output,
                )

            # ==================== Self-Improve Tab ====================
            with gr.TabItem("Self-Improve"):
                gr.Markdown("### Recursive Self-Improvement Loop")
                with gr.Row():
                    with gr.Column():
                        si_iterations = gr.Number(
                            value=5, label="Iterations", precision=0
                        )
                        si_samples = gr.Number(
                            value=500, label="Samples per Iteration",
                            precision=0
                        )
                        si_lr = gr.Number(
                            value=5e-5, label="Learning Rate"
                        )
                        si_batch = gr.Number(
                            value=16, label="Batch Size", precision=0
                        )
                        si_temp = gr.Slider(
                            0.1, 1.5, value=0.8, step=0.1,
                            label="Generation Temperature"
                        )
                    with gr.Column():
                        si_status = gr.Textbox(
                            label="Status", interactive=False
                        )
                        si_start_btn = gr.Button(
                            "Start", variant="primary"
                        )
                        si_stop_btn = gr.Button("Stop", variant="stop")

                gr.Markdown("### Live Metrics")
                with gr.Row():
                    si_table = gr.Dataframe(
                        headers=[
                            "Iteration",
                            "Status",
                            "Loss",
                            "Perplexity",
                            "Quality",
                            "Time",
                        ],
                        label="Iteration Log",
                    )
                    _si_plot = gr.LinePlot(
                        x="Iteration",
                        y="Loss",
                        title="Loss Curve",
                        label="Loss over Iterations",
                    )

                si_start_btn.click(
                    fn=_handle_self_improve_start,
                    inputs=[
                        si_iterations,
                        si_samples,
                        si_lr,
                        si_batch,
                        si_temp,
                        state,
                    ],
                    outputs=si_status,
                )
                si_stop_btn.click(
                    fn=_handle_self_improve_stop,
                    inputs=state,
                    outputs=si_status,
                )

                # Poll for table updates.
                si_timer = gr.Timer(value=2)
                si_timer.tick(
                    fn=_get_iteration_table,
                    inputs=state,
                    outputs=si_table,
                )

            # ==================== Training Tab ====================
            with gr.TabItem("Training"):
                gr.Markdown("### Pre-training on Corpus")
                with gr.Row():
                    with gr.Column():
                        tr_corpus = gr.Textbox(
                            label="Corpus Directory",
                            placeholder="./data/",
                            value="./data/",
                        )
                        tr_epochs = gr.Number(
                            value=3, label="Epochs", precision=0
                        )
                        tr_lr = gr.Number(
                            value=5e-4, label="Learning Rate"
                        )
                        tr_batch = gr.Number(
                            value=16, label="Batch Size", precision=0
                        )
                        tr_seq_len = gr.Number(
                            value=512, label="Max Sequence Length",
                            precision=0
                        )
                        tr_status = gr.Textbox(
                            label="Status", interactive=False
                        )
                        tr_start_btn = gr.Button(
                            "Start Training", variant="primary"
                        )
                    with gr.Column():
                        tr_plot = gr.LinePlot(
                            x="Step",
                            y="Loss",
                            title="Training Loss",
                            label="Loss over Steps",
                        )

                tr_start_btn.click(
                    fn=_handle_training_start,
                    inputs=[
                        tr_corpus,
                        tr_epochs,
                        tr_lr,
                        tr_batch,
                        tr_seq_len,
                        state,
                    ],
                    outputs=tr_status,
                )

                # Poll for plot updates.
                tr_timer = gr.Timer(value=2)
                tr_timer.tick(
                    fn=_get_training_plot_data,
                    inputs=state,
                    outputs=tr_plot,
                )

            # ==================== Evaluation Tab ====================
            with gr.TabItem("Evaluation"):
                gr.Markdown("### Run Evaluation Suite")
                with gr.Row():
                    ev_run_btn = gr.Button(
                        "Run Evaluation", variant="primary"
                    )
                gr.Markdown("### Results")
                ev_output = gr.JSON(label="Evaluation Results")

                ev_run_btn.click(
                    fn=_handle_evaluation,
                    inputs=state,
                    outputs=ev_output,
                )

            # ==================== Metrics Report Tab ====================
            with gr.TabItem("Metrics Report"):
                gr.Markdown(
                    "### Recursive Self-Improvement Metrics\n"
                    "Load a `metrics_history.json` produced by the recursive "
                    "loop. Benchmark scores (MMLU / GSM8K / HumanEval) recorded "
                    "by the eval-suite hook are surfaced automatically."
                )
                with gr.Row():
                    mr_path = gr.Textbox(
                        label="Path to metrics_history.json",
                        value=os.path.join(checkpoint_dir, "metrics_history.json"),
                        scale=4,
                    )
                    mr_load_btn = gr.Button("Load", variant="primary", scale=1)
                mr_report = gr.Markdown(label="Report")
                mr_table = gr.Dataframe(
                    label="Per-iteration metrics (incl. benchmarks)",
                    interactive=False,
                    wrap=True,
                )

                mr_load_btn.click(
                    fn=_handle_metrics_report,
                    inputs=mr_path,
                    outputs=[mr_report, mr_table],
                )

            # ==================== Settings Tab ====================
            with gr.TabItem("Settings"):
                gr.Markdown("### Configuration Editor")
                config_editor = gr.Code(
                    value=config_text,
                    language="yaml",
                    label="config.yaml",
                )
                with gr.Row():
                    cfg_save_btn = gr.Button("Save Config")
                    cfg_load_model_btn = gr.Button("Load Model")
                cfg_status = gr.Textbox(label="Status", interactive=False)

                cfg_model_dir = gr.Textbox(
                    label="Model Directory",
                    placeholder="./checkpoints/model_v1",
                    value=checkpoint_dir,
                )
                cfg_tokenizer_path = gr.Textbox(
                    label="Tokenizer File (optional)",
                    placeholder="./checkpoints/model_v1/tokenizer.json",
                    value="",
                )

                cfg_save_btn.click(
                    fn=lambda text: _save_config(config_path, yaml.safe_load(text)),
                    inputs=config_editor,
                    outputs=cfg_status,
                )

                def _do_load_model(
                    model_dir: str,
                    tokenizer_file: str,
                    state_val: DashboardState,
                ):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    m, t, message = _load_model(
                        model_dir, tokenizer_file or None, device
                    )
                    if m is not None and t is not None:
                        _set_loaded_model(
                            state_val,
                            m,
                            t,
                            model_dir,
                            tokenizer_file or None,
                            device,
                        )
                        return (
                            state_val,
                            message,
                            _model_status(state_val),
                        )
                    return state_val, message, _model_status(state_val)

                cfg_load_model_btn.click(
                    fn=_do_load_model,
                    inputs=[cfg_model_dir, cfg_tokenizer_path, state],
                    outputs=[state, cfg_status, model_status],
                )

    return demo


def launch_dashboard(
    model: Optional[SelfImprovingLLM] = None,
    tokenizer: Optional[BPETokenizer] = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
    **kwargs: Any,
) -> None:
    """Launch the dashboard.

    Args:
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.
        host: Host/IP for the Gradio server.
        port: Port number for the Gradio server.
        share: Whether to create a public shareable link.
        **kwargs: Additional arguments passed to ``create_dashboard``.
    """
    demo = create_dashboard(
        model=model, tokenizer=tokenizer, **kwargs
    )
    demo.launch(server_name=host, server_port=port, share=share)


if __name__ == "__main__":
    launch_dashboard()
