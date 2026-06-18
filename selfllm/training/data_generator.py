"""Self-training data generation via model self-prompting."""
import logging
import random
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from selfllm.model.model import SelfImprovingLLM
from selfllm.model.quality_model import QualityModel
from selfllm.model.tokenizer import BPETokenizer

logger = logging.getLogger(__name__)


class SelfTrainingDataGenerator:
    """Generates training data by prompting the model with itself.

    Uses diverse prompt templates and seed topics to generate varied training
    samples. Implements a "self-critic" approach: generates multiple responses
    per prompt, scores each with the quality model, and keeps the best one.
    """

    # 10 diverse prompt templates covering academic, creative, analytical,
    # and problem-solving domains.
    PROMPT_TEMPLATES: List[str] = [
        "Explain the concept of {topic} in detail:",
        "Write a creative story about {topic}:",
        "Compare and contrast {topic_a} and {topic_b}:",
        "Solve this problem step by step: {problem}",
        "Provide a detailed analysis of {topic}:",
        "Write a dialogue between characters discussing {topic}:",
        "Describe the process of {process}:",
        "What are the implications of {topic}? Explain thoroughly:",
        "Write an essay on the importance of {topic}:",
        "Generate a list of ideas related to {topic} with explanations:",
    ]

    # 100+ diverse seed topics spanning many domains.
    SEED_TOPICS: List[str] = [
        # Science and Technology
        "artificial intelligence", "quantum computing", "machine learning",
        "neural networks", "blockchain technology", "climate change",
        "renewable energy", "space exploration", "genetic engineering",
        "nanotechnology", "virtual reality", "cybersecurity",
        "robotics", "big data analytics", "cloud computing",
        "internet of things", "5G networks", "biotechnology",
        "fusion energy", "carbon nanotubes",
        # Mathematics
        "linear algebra", "calculus", "number theory",
        "graph theory", "probability and statistics", "differential equations",
        "topology", "set theory", "abstract algebra",
        "mathematical optimization",
        # Philosophy
        "epistemology", "ethics", "metaphysics",
        "philosophy of mind", "existentialism", "stoicism",
        "free will", "the nature of consciousness", "moral philosophy",
        "philosophy of science",
        # History
        "the Renaissance", "the Industrial Revolution", "ancient Rome",
        "the Silk Road", "the French Revolution", "the Cold War",
        "the Age of Exploration", "the Scientific Revolution",
        "the Byzantine Empire", "the Mongol Empire",
        # Arts and Literature
        "impressionism", "modernist poetry", "Shakespearean drama",
        "classical music composition", "film noir", "surrealism",
        "Greek mythology", "Japanese haiku", "the novel as an art form",
        "digital art",
        # Social Sciences
        "behavioral economics", "social psychology", "political theory",
        "cultural anthropology", "urban planning", "education policy",
        "criminal justice reform", "public health systems",
        "international relations", "demographic shifts",
        # Nature and Environment
        "ecosystem biodiversity", "ocean currents", "volcanic activity",
        "the water cycle", "photosynthesis", "evolutionary adaptation",
        "food webs", "atmospheric dynamics", "geological time scales",
        "sustainable agriculture",
        # Problem-solving / Processes
        "designing a database schema", "training a neural network",
        "optimizing supply chains", "debugging complex software",
        "planning a research study", "building a recommendation system",
        "conducting a literature review", "analyzing financial markets",
        "creating a marketing strategy", "performing risk assessment",
        # Abstract / Creative topics
        "the concept of time", "the nature of beauty",
        "the meaning of progress", "the role of imagination",
        "the relationship between language and thought",
        "the ethics of artificial intelligence",
        "the future of human civilization",
        "the intersection of art and technology",
        "the psychology of creativity", "the philosophy of mathematics",
        "emergence in complex systems", "information theory",
        "swarm intelligence", "recursive self-reference",
    ]

    def __init__(
        self,
        model: SelfImprovingLLM,
        tokenizer: BPETokenizer,
        quality_model: Optional[QualityModel] = None,
        device: str = "cuda",
    ) -> None:
        """Initialize the data generator.

        Args:
            model: The SelfImprovingLLM to use for generation.
            tokenizer: BPETokenizer for encoding prompts and decoding responses.
            quality_model: Optional QualityModel for scoring responses.
            device: Device to run generation on ('cuda' or 'cpu').
        """
        self.model = model
        self.tokenizer = tokenizer
        self.quality_model = quality_model
        self.device = device

    def generate_prompt(self, topic: Optional[str] = None) -> str:
        """Generate a diverse prompt by selecting a template and filling it.

        Args:
            topic: Optional specific topic to use. If None, a random topic
                is selected from SEED_TOPICS.

        Returns:
            A formatted prompt string ready for generation.
        """
        if topic is None:
            topic = random.choice(self.SEED_TOPICS)

        template = random.choice(self.PROMPT_TEMPLATES)

        # Fill template placeholders
        if "{topic_a}" in template and "{topic_b}" in template:
            # Need two different topics
            topics = random.sample(self.SEED_TOPICS, 2)
            prompt = template.replace("{topic_a}", topics[0]).replace("{topic_b}", topics[1])
        elif "{problem}" in template:
            prompt = template.replace("{problem}", topic)
        elif "{process}" in template:
            prompt = template.replace("{process}", topic)
        elif "{topic}" in template:
            prompt = template.replace("{topic}", topic)
        else:
            prompt = template

        return prompt

    def generate_response_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.92,
        batch_size: int = 8,
    ) -> List[Dict[str, Any]]:
        """Generate responses for a batch of prompts.

        Handles CUDA batching to avoid OOM errors. Returns metadata-rich
        results including generation confidence scores.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            batch_size: Number of prompts to process per forward pass.

        Returns:
            List of dicts with keys:
                - prompt: str
                - response: str
                - token_ids: List[int] (full sequence)
                - generation_metadata: dict with temperature, top_p, length,
                  confidence_score
        """
        results: List[Dict[str, Any]] = []

        num_batches = (len(prompts) + batch_size - 1) // batch_size

        for i in tqdm(range(num_batches), desc="Generating responses", leave=False):
            batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]

            try:
                # Encode all prompts in the batch
                batch_token_ids = [self.tokenizer.encode(p) for p in batch_prompts]

                # Pad to same length for batching
                max_prompt_len = max(len(ids) for ids in batch_token_ids)
                padded_ids = []
                for ids in batch_token_ids:
                    padded = ids + [self.tokenizer.pad_token_id] * (
                        max_prompt_len - len(ids)
                    )
                    padded_ids.append(padded)

                prompt_tensor = torch.tensor(padded_ids, device=self.device)

                # Generate
                with torch.no_grad():
                    gen_output = self.model.generate(
                        prompt_ids=prompt_tensor,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        return_hidden=False,
                    )

                sequences = gen_output["sequences"]
                scores = gen_output.get("scores", None)

                # Process each generated sequence
                for j, prompt_text in enumerate(batch_prompts):
                    seq = sequences[j].tolist()
                    prompt_len = len(batch_token_ids[j])

                    # Remove prompt portion from sequence
                    full_seq = seq
                    response_ids = seq[prompt_len:]

                    # Strip padding and EOS from response
                    response_ids = [
                        t
                        for t in response_ids
                        if t != self.tokenizer.pad_token_id
                    ]

                    # Decode response
                    response_text = self.tokenizer.decode(response_ids)

                    # Compute confidence score from generation scores if available
                    confidence = 0.5
                    if scores is not None and len(scores) > j:
                        try:
                            conf_vals = torch.stack(scores[j]).cpu()
                            confidence = torch.mean(conf_vals).item()
                        except Exception:
                            confidence = 0.5

                    metadata = {
                        "temperature": temperature,
                        "top_p": top_p,
                        "length": len(response_ids),
                        "confidence_score": float(confidence),
                    }

                    results.append(
                        {
                            "prompt": prompt_text,
                            "response": response_text,
                            "token_ids": full_seq,
                            "generation_metadata": metadata,
                        }
                    )

            except torch.cuda.OutOfMemoryError:
                logger.warning(
                    f"CUDA OOM on batch {i}. Clearing cache and retrying with smaller batch."
                )
                torch.cuda.empty_cache()
                # Retry one by one
                for prompt_text in batch_prompts:
                    try:
                        prompt_ids = self.tokenizer.encode(prompt_text)
                        prompt_tensor = torch.tensor([prompt_ids], device=self.device)
                        with torch.no_grad():
                            gen_output = self.model.generate(
                                prompt_ids=prompt_tensor,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p,
                                return_hidden=False,
                            )
                        seq = gen_output["sequences"][0].tolist()
                        prompt_len = len(prompt_ids)
                        response_ids = seq[prompt_len:]
                        response_ids = [
                            t
                            for t in response_ids
                            if t != self.tokenizer.pad_token_id
                        ]
                        response_text = self.tokenizer.decode(response_ids)
                        metadata = {
                            "temperature": temperature,
                            "top_p": top_p,
                            "length": len(response_ids),
                            "confidence_score": 0.5,
                        }
                        results.append(
                            {
                                "prompt": prompt_text,
                                "response": response_text,
                                "token_ids": seq,
                                "generation_metadata": metadata,
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to generate for prompt: {e}")
                        results.append(
                            {
                                "prompt": prompt_text,
                                "response": "",
                                "token_ids": [],
                                "generation_metadata": {
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "length": 0,
                                    "confidence_score": 0.0,
                                    "error": str(e),
                                },
                            }
                        )
            except Exception as e:
                logger.error(f"Batch generation failed: {e}")
                # Add empty results for failed batch
                for prompt_text in batch_prompts:
                    results.append(
                        {
                            "prompt": prompt_text,
                            "response": "",
                            "token_ids": [],
                            "generation_metadata": {
                                "temperature": temperature,
                                "top_p": top_p,
                                "length": 0,
                                "confidence_score": 0.0,
                                "error": str(e),
                            },
                        }
                    )

        return results

    def _score_response(self, response: str) -> float:
        """Score a single response using the quality model.

        Args:
            response: Response text to score.

        Returns:
            Quality score (higher is better). Returns 0.5 if quality model
            is not available.
        """
        if self.quality_model is None or not response:
            return 0.5

        try:
            token_ids = self.tokenizer.encode(response)
            if not token_ids:
                return 0.0
            token_ids = token_ids[:512]  # Truncate
            ids_tensor = torch.tensor([token_ids], device=self.device)

            with torch.no_grad():
                q_out = self.quality_model(ids_tensor)
                score = q_out["overall_score"][0].item()
                return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.warning(f"Quality scoring failed: {e}")
            return 0.5

    def generate_training_batch(
        self,
        num_samples: int = 1000,
        num_critic_samples: int = 3,
    ) -> List[Dict[str, Any]]:
        """Main method: Generate a batch of self-training data.

        Uses the "self-critic" approach: for each prompt, generates
        ``num_critic_samples`` responses, scores each with the quality model,
        and keeps only the highest-quality response.

        Args:
            num_samples: Number of prompts to generate responses for.
            num_critic_samples: Number of candidate responses to generate per
                prompt (self-critic N-best selection).

        Returns:
            List of sample dicts with keys 'prompt', 'response',
            'token_ids', 'generation_metadata', and 'quality_score'.
        """
        logger.info(
            f"Generating training batch: {num_samples} prompts x "
            f"{num_critic_samples} candidates = "
            f"{num_samples * num_critic_samples} total generations"
        )

        # Step 1: Generate diverse prompts
        prompts = [self.generate_prompt() for _ in range(num_samples)]

        # Step 2: For self-critic, repeat each prompt num_critic_samples times
        expanded_prompts = []
        prompt_indices = []
        for idx, prompt in enumerate(prompts):
            for _ in range(num_critic_samples):
                expanded_prompts.append(prompt)
                prompt_indices.append(idx)

        # Step 3: Generate all responses in batches
        all_results = self.generate_response_batch(expanded_prompts)

        # Step 4: Group by prompt and select best response
        best_samples: List[Dict[str, Any]] = []

        # Group results by original prompt index
        grouped: Dict[int, List[Dict[str, Any]]] = {}
        for pidx, result in zip(prompt_indices, all_results):
            grouped.setdefault(pidx, []).append(result)

        for pidx in tqdm(range(num_samples), desc="Self-critic scoring"):
            candidates = grouped.get(pidx, [])
            if not candidates:
                continue

            # Filter out empty/error responses
            valid_candidates = [
                c for c in candidates if c.get("response") and not c.get("generation_metadata", {}).get("error")
            ]
            if not valid_candidates:
                valid_candidates = candidates

            # Score each candidate with quality model
            scored_candidates = []
            for cand in valid_candidates:
                quality_score = self._score_response(cand["response"])
                scored_candidates.append((quality_score, cand))

            # Select highest quality response
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_candidate = scored_candidates[0]

            best_sample = dict(best_candidate)
            best_sample["quality_score"] = float(best_score)
            best_sample["num_candidates"] = len(valid_candidates)
            best_samples.append(best_sample)

        logger.info(f"Generated {len(best_samples)} training samples "
                     f"(selected best from {num_critic_samples} candidates each)")

        return best_samples
