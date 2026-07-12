"""User profile configuration for personalized LLM training."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class UserProfile:
    """Personalization profile describing the target user and style.

    Attributes:
        name: Display name for the personalized model.
        topics: Subject areas the model should specialize in.
        style: Writing / response style description.
        eval_prompts: Prompts used to evaluate and steer self-improvement.
        system_prompt: Optional system prompt baked into the model manifest.
    """

    name: str = "User"
    topics: List[str] = field(default_factory=list)
    style: str = "clear, helpful, and concise"
    eval_prompts: List[str] = field(default_factory=list)
    system_prompt: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the profile to a plain dictionary."""
        return {
            "name": self.name,
            "topics": list(self.topics),
            "style": self.style,
            "eval_prompts": list(self.eval_prompts),
            "system_prompt": self.system_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Build a profile from a dictionary."""
        return cls(
            name=str(data.get("name", "User")),
            topics=list(data.get("topics", [])),
            style=str(data.get("style", "clear, helpful, and concise")),
            eval_prompts=list(data.get("eval_prompts", [])),
            system_prompt=str(data.get("system_prompt", "")),
        )

    def default_eval_prompts(self) -> List[str]:
        """Return profile eval prompts, with topic-based fallbacks."""
        if self.eval_prompts:
            return list(self.eval_prompts)

        prompts: List[str] = []
        for topic in self.topics[:5]:
            prompts.append(f"Explain {topic} in your own words.")
            prompts.append(f"What is the most important thing to know about {topic}?")

        if self.name and self.name != "User":
            prompts.append(f"Write a short introduction as {self.name} would.")
            prompts.append(f"How would you describe your expertise to a colleague?")

        if not prompts:
            prompts = [
                "Summarize your main areas of expertise.",
                "Write a short note in your preferred style.",
                "Explain a concept you care about clearly.",
            ]
        return prompts


def load_profile(path: Optional[str]) -> UserProfile:
    """Load a user profile from YAML, or return defaults when *path* is None."""
    if not path:
        return UserProfile()

    profile_path = Path(path)
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    with open(profile_path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    profile_data = data.get("profile", data)
    return UserProfile.from_dict(profile_data)
