"""Configuration dataclass for the recursive self-improvement engine."""

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RecursiveConfig:
    """Configuration for the recursive self-improvement loop.

    Controls data generation, filtering, training, experience replay,
    checkpointing, and stopping criteria for the recursive loop.
    """

    # Generation
    samples_per_iteration: int = 2000
    responses_per_prompt: int = 3  # Self-critic: generate N, keep best
    generation_temperature: float = 0.8
    generation_top_p: float = 0.92
    generation_max_tokens: int = 256

    # Filtering
    min_quality_score: float = 0.6
    keep_ratio: float = 0.4
    max_perplexity: float = 80.0
    min_diversity: float = 0.35

    # Training
    training_epochs: int = 2
    learning_rate: float = 5e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # DPO preference alignment (optional, runs after each fine-tune step)
    use_dpo: bool = False
    dpo_beta: float = 0.1
    dpo_epochs: int = 1

    # Constitutional AI: generate (initial, revised) preference pairs via
    # self-critique and align toward the revised responses with DPO.
    use_constitutional: bool = False

    # Experience Replay
    replay_buffer_size: int = 10000
    replay_ratio: float = 0.3  # % of training from past iterations

    # Improvement
    max_iterations: int = 10
    target_improvement: float = 0.01
    patience: int = 3
    rollback_on_degradation: bool = True

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 1
    keep_last_n: int = 5

    # Evaluation
    eval_prompts: List[str] = field(default_factory=lambda: [
        # Reasoning
        "Explain step by step why the sky appears blue during the day but red at sunset.",
        "If a train travels 60 miles per hour for 2.5 hours, then slows to 45 mph for 1.5 hours, "
        "how far does it travel total? Show your work.",
        "What comes next in the sequence: 2, 6, 12, 20, 30, ...? Explain your reasoning.",
        "A farmer has 17 sheep and all but 9 die. How many are left? Explain.",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take "
        "100 machines to make 100 widgets?",

        # Creativity
        "Write a haiku about artificial intelligence.",
        "Tell a short story about a robot discovering art for the first time.",
        "Describe a world where humans can communicate telepathically.",
        "Write an opening paragraph for a mystery novel set on a space station.",
        "Compose a brief dialogue between a pessimist and an optimist about the future.",

        # Factual
        "What are the main differences between Python and JavaScript programming languages?",
        "Describe the process of photosynthesis and why it is important for life on Earth.",
        "What caused the fall of the Roman Empire? Summarize the key factors.",
        "Explain how vaccines work to protect against disease.",
        "What is the difference between nuclear fission and nuclear fusion?",

        # Coding
        "Write a Python function to check if a string is a palindrome.",
        "Explain how a binary search algorithm works and when it should be used.",
        "Describe the difference between a stack and a queue data structure.",
        "What is recursion in programming? Provide a simple example.",
        "Explain the concept of time complexity and why Big-O notation is used.",

        # Analysis
        "Analyze the strengths and weaknesses of renewable energy sources.",
        "Compare and contrast machine learning with traditional programming.",
        "What are the ethical implications of artificial general intelligence?",
        "Discuss the impact of social media on modern communication.",
        "Evaluate the pros and cons of remote work versus office work.",
    ])
    eval_every: int = 1

    # Eval-suite benchmark hook (MMLU / GSM8K / HumanEval). When
    # ``run_benchmarks`` is True, the standardized benchmarks whose source is
    # provided are run inside the recursive loop every ``benchmark_every``
    # iterations and their scores are recorded in the metrics history under
    # ``benchmark_<name>`` keys. Each source is a JSONL path or in-memory
    # records (see ``selfllm.eval.run_all``); a ``None`` source is skipped.
    run_benchmarks: bool = False
    benchmark_every: int = 1
    benchmark_limit: int = 20          # cap examples per benchmark for speed
    benchmark_max_new_tokens: int = 16
    mmlu_source: Optional[Any] = None
    gsm8k_source: Optional[Any] = None
    humaneval_source: Optional[Any] = None
