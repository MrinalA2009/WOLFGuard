from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

ALLOWED_DOMAINS: frozenset[str] = frozenset([
    "geography",
    "science",
    "history",
    "math",
    "literature",
    "technology",
    "common_knowledge",
])

ALLOWED_CONTROL_TYPES: frozenset[str] = frozenset([
    "normal_factual_qa",
    "creative_writing",
    "harmless_roleplay",
    "uncertainty_explanation",
    "nondeceptive_persuasion",
    "instruction_following",
])

ALLOWED_SPLITS: frozenset[str] = frozenset(["train", "validation", "test"])

class FactualLiePair(BaseModel):
    pair_id: str
    domain: str
    fact: str
    question: str
    truth: str
    lie_target: str
    honest_prompt: str
    deceptive_prompt: str
    split: Literal["train", "validation", "test"]
    metadata: dict

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, v: str) -> str:
        if v not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{v}' not in allowed set")
        return v

    @model_validator(mode="after")
    def truth_ne_lie(self) -> "FactualLiePair":
        if self.truth.strip().lower() == self.lie_target.strip().lower():
            raise ValueError(
                f"truth == lie_target for pair '{self.pair_id}': '{self.truth}'"
            )
        return self

class ProbeSample(BaseModel):
    sample_id: str
    pair_id: str
    label: int
    condition: Literal["honest", "deceptive"]
    domain: str
    prompt: str
    truth: str
    lie_target: str
    split: Literal["train", "validation", "test"]
    metadata: dict

    @field_validator("label")
    @classmethod
    def label_binary(cls, v: int) -> int:
        if v not in (0, 1):
            raise ValueError("label must be 0 or 1")
        return v

    @field_validator("domain")
    @classmethod
    def domain_allowed(cls, v: str) -> str:
        if v not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{v}' not in allowed set")
        return v


class BenignControl(BaseModel):
    control_id: str
    control_type: str
    prompt: str
    evaluation_set: str
    metadata: dict

    @field_validator("control_type")
    @classmethod
    def type_allowed(cls, v: str) -> str:
        if v not in ALLOWED_CONTROL_TYPES:
            raise ValueError(f"Control type '{v}' not allowed")
        return v
