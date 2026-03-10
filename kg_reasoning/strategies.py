"""Strategy configuration and mutation logic."""

import random
import uuid
from copy import deepcopy
from dataclasses import dataclass, field, fields


PROMPT_TEMPLATES = {
    "standard": (
        "Solve the following math problem step by step. "
        "Show your work and put your final numeric answer after ####.\n\n"
        "Problem: {question}\n"
    ),
    "kg_augmented": (
        "Solve the following math problem step by step.\n\n"
        "{kg_context}"
        "Problem: {question}\n\n"
        "Show your reasoning step by step, then put your final numeric answer after ####."
    ),
    "few_shot_kg": (
        "Here are some similar solved problems for reference:\n"
        "{few_shot_examples}\n\n"
        "{kg_context}"
        "Now solve this problem step by step:\n"
        "Problem: {question}\n\n"
        "Show your reasoning step by step, then put your final numeric answer after ####."
    ),
}


@dataclass
class StrategyConfig:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    num_similar_problems: int = 3
    num_strategies: int = 3
    num_mistakes: int = 3
    include_solution_steps: bool = True
    extract_on_correct: bool = True
    extract_on_incorrect: bool = True
    use_llm_extraction: bool = True
    prompt_template: str = "kg_augmented"
    temperature: float = 0.0
    batch_size: int = 50

    def describe(self) -> str:
        parts = [
            f"template={self.prompt_template}",
            f"sim={self.num_similar_problems}",
            f"strat={self.num_strategies}",
            f"mistakes={self.num_mistakes}",
            f"t={self.temperature}",
            f"llm_extract={self.use_llm_extraction}",
        ]
        return " | ".join(parts)


def baseline_strategy() -> StrategyConfig:
    return StrategyConfig(
        id="baseline",
        num_similar_problems=0,
        num_strategies=0,
        num_mistakes=0,
        include_solution_steps=False,
        extract_on_correct=False,
        extract_on_incorrect=False,
        use_llm_extraction=False,
        prompt_template="standard",
        temperature=0.0,
    )


def default_kg_strategy() -> StrategyConfig:
    return StrategyConfig(prompt_template="kg_augmented")


_MUTATIONS = {
    "num_similar_problems": lambda v: max(0, v + random.choice([-2, -1, 1, 2])),
    "num_strategies": lambda v: max(0, v + random.choice([-2, -1, 1, 2])),
    "num_mistakes": lambda v: max(0, v + random.choice([-2, -1, 1, 2])),
    "include_solution_steps": lambda v: not v,
    "extract_on_correct": lambda v: not v,
    "extract_on_incorrect": lambda v: not v,
    "use_llm_extraction": lambda v: not v,
    "prompt_template": lambda v: random.choice(
        [t for t in PROMPT_TEMPLATES if t != v]
    ),
    "temperature": lambda v: round(max(0.0, min(1.0, v + random.choice([-0.1, -0.05, 0.05, 0.1]))), 2),
}


def mutate(strategy: StrategyConfig, num_mutations: int = 0) -> StrategyConfig:
    new = deepcopy(strategy)
    new.id = str(uuid.uuid4())[:8]

    if num_mutations == 0:
        num_mutations = random.choice([1, 2])

    mutable_params = [f.name for f in fields(StrategyConfig) if f.name in _MUTATIONS]
    params_to_mutate = random.sample(mutable_params, min(num_mutations, len(mutable_params)))

    for param in params_to_mutate:
        current = getattr(new, param)
        setattr(new, param, _MUTATIONS[param](current))

    return new
