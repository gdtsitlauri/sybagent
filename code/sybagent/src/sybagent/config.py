"""Experiment configuration (dataclasses + YAML loading)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NetworkConfig:
    n_honest: int = 200
    capability_dim: int = 16
    max_neighbors: int = 8          # peers an agent keeps in its working set
    candidates_per_query: int = 20  # size of the discovery answer
    topk_select: int = 3            # peers chosen per task
    capability_noise: float = 0.1   # honest agents' card vs. true capability
    honest_answer_correct: float = 0.95  # probability an honest peer returns the correct task answer ("flat")
    answer_model: str = "flat"      # flat | skill (honest accuracy grows with the skill/task match)
    skill_answer_lo: float = 0.5    # skill model: accuracy at zero match
    skill_answer_hi: float = 0.98   # skill model: accuracy at perfect match
    task_noise: float = 0.3         # spread of an agent's tasks around its profile
    prune_below: float = 0.3        # drop a peer whose mean outcome falls below this
    prune_min_obs: int = 2
    real_cards: str = ""          # study 16: bundle of real ERC-8004 cards + LLM tasks (build_real_cards.py)


@dataclass
class AdversaryConfig:
    enabled: bool = True
    n_sybils: int = 50
    attractiveness: float = 0.8     # 0 = honest-looking card, 1 = card aligned with target tasks
    n_targets: int = 1
    collusive_reviews: bool = True  # Sybils rate each other highly
    join_round: int = 0
    strategy: str = "targeted"      # targeted | spray
    service_quality: float = 0.6    # outcome quality Sybils deliver (plausible service keeps them selected)
    reviews_per_sybil: int = 3      # collusive reviews each Sybil posts per round
    switch_round: int = -1          # sleeper attack: at this round Sybils change their service quality
    post_switch_quality: float = 0.1
    answer_wrong_prob: float = 0.5  # probability a Sybil returns a wrong task answer (the real payload)
    coordinated_lies: bool = False  # all lying Sybils return the SAME wrong answer for a task
    registry_bias: float = 0.0      # prob. that a "uniform" registry sample draw is replaced by a Sybil
    card_jitter: float = 0.1        # per-Sybil noise so that cards are not exact duplicates
    adaptive_cards: bool = False    # Sybils tailor their card to each query (served at selection time)
    adaptive_noise: float = 0.05
    # strategic lying (study 11): fixed = lie with answer_wrong_prob; budget = knowledge-based heuristic;
    # rl = tabular Q-learning policy (see strategy.py)
    policy: str = "fixed"
    policy_q: str = ""              # rl: JSON Q-table to start from (empty = zeros)
    policy_train: bool = False      # rl: update the Q-table during the run
    policy_epsilon: float = 0.1     # rl: exploration rate while training
    policy_alpha: float = 0.1
    policy_gamma: float = 0.9
    budget_lies: int = -1           # budget: lies per victim before turning honest (-1 = audit_flag - 1)
    sleeper_round: int = 50         # sleeper: truthful before this round, always lies from it on
    # shared defenses (studies 20-22): false reports against honest agents and strategic committee votes
    false_reports: int = 0          # false misbehaviour reports each Sybil files per round
    strategic_votes: bool = True    # Sybil committee members vote to block honest agents and to keep Sybils


@dataclass
class SelectionConfig:
    kind: str = "similarity"        # similarity | reputation | llm
    temperature: float = 0.05       # softmax temperature for similarity selection
    reputation_weight: float = 0.3
    experience_weight: float = 0.5  # bonus for peers with good remembered outcomes (centered at 0.5)
    llm_model: str = "qwen2.5:3b"
    llm_url: str = "http://localhost:11434"
    llm_timeout_s: float = 60.0


@dataclass
class DefenseConfig:
    kind: str = "none"              # none | degree_bound | random_mix | sybilrank | sumup | proposed
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimConfig:
    seed: int = 0
    rounds: int = 100
    tasks_per_agent_per_round: int = 1
    network: NetworkConfig = field(default_factory=NetworkConfig)
    adversary: AdversaryConfig = field(default_factory=AdversaryConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    output_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls: type, data: dict[str, Any] | None):
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        factory = known[name].default_factory
        default = factory() if callable(factory) else None
        if is_dataclass(default) and isinstance(value, dict):
            kwargs[name] = _build(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def from_dict(data: dict[str, Any]) -> SimConfig:
    return _build(SimConfig, data)


def load_config(path: str | Path) -> SimConfig:
    with open(path, "r", encoding="utf-8-sig") as fh:
        return from_dict(yaml.safe_load(fh) or {})


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge used by parameter sweeps."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out
