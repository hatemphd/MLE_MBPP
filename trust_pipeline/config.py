from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

# Rows that never enter calibration/test splits or the sibling-agreement pool (outside SMOKE_TEST):
# mutated reference solutions and LLM samples that were asked to contain a bug.
TRAINING_ONLY_PROVENANCE = {"mutant", "injected"}


def default_model_configs(use_local_hf=True, use_openai=False, use_anthropic=False):
    """Model/setting combos. Sampling (temperature > 0) is required for self-consistency features."""
    configs = []
    if use_local_hf:
        configs += [
            {"name": "qwen0.5b-t0.3", "provider": "local_hf",
             "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "temperature": 0.3},
            {"name": "qwen0.5b-t1.0", "provider": "local_hf",
             "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "temperature": 1.0},
            {"name": "qwen1.5b-t0.6", "provider": "local_hf",
             "model": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "temperature": 0.6},
        ]
    if use_openai:
        configs += [
            {"name": "gpt-4o-mini-t0.3", "provider": "openai", "model": "gpt-4o-mini", "temperature": 0.3},
            {"name": "gpt-4o-mini-t1.0", "provider": "openai", "model": "gpt-4o-mini", "temperature": 1.0},
        ]
    if use_anthropic:
        configs += [
            {"name": "claude-haiku-t0.7", "provider": "anthropic",
             "model": "claude-haiku-4-5-20251001", "temperature": 0.7},
        ]
    return configs


@dataclass
class Config:
    seed: int = 42

    # Run mode: SMOKE_TEST uses reference solutions + AST mutants instead of an LLM.
    smoke_test: bool = False
    smoke_mutants_per_problem: int = 7

    # Dataset size
    num_problems: Optional[int] = 100       # None -> all 974 MBPP problems
    samples_per_config: int = 5
    probes_per_problem: int = 6
    use_mutant_augmentation: bool = False   # mutants as extra TRAINING rows only
    mutants_per_problem: int = 3

    # Execution
    max_workers: int = field(default_factory=lambda: max(2, os.cpu_count() or 2))
    item_timeout: float = 3.0
    memory_limit_mb: int = 1024

    # Human validation
    human_added_feedback: bool = True       # True = simulated reviewer (demo only)
    human_sample_size: int = 150

    # Trust decision
    target_approve_precision: float = 0.95
    target_reject_precision: float = 0.95
    min_decision_support: int = 10
    conformal_alpha: float = 0.10

    # Generation backends
    use_local_hf: bool = True
    use_openai: bool = False
    use_anthropic: bool = False
    max_new_tokens: int = 512
    model_configs: list = field(default_factory=list)   # empty -> default_model_configs(...)
    # Optional extra: prompt styles crossed with every model config (see generation.PROMPT_STYLES).
    # ["standard"] is the core design; "inject_bug" samples are training-only.
    prompt_styles: list = field(default_factory=lambda: ["standard"])

    output_dir: str = "artifacts"

    # Run report
    bootstrap_samples: int = 1000           # resamples (by problem) for test-metric confidence intervals
    history_file: Optional[str] = None      # None -> runs_history.csv in the parent of output_dir

    def resolved_model_configs(self):
        if self.smoke_test:
            return [{"name": "smoke", "provider": "smoke"}]
        if self.model_configs:
            return self.model_configs
        return default_model_configs(self.use_local_hf, self.use_openai, self.use_anthropic)

    def resolved_prompt_styles(self):
        from .generation import resolve_prompt_styles
        return resolve_prompt_styles(self.prompt_styles)

    @property
    def model_names(self):
        return [c["name"] for c in self.resolved_model_configs()]

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
