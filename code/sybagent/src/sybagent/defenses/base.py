"""Defense interface: hooks on the candidate list and at the end of each round."""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent
from sybagent.config import DefenseConfig
from sybagent.context import SimContext


class Defense:
    """No-op defense; subclasses override the hooks they need."""

    name = "none"

    def __init__(self, cfg: DefenseConfig) -> None:
        self.cfg = cfg
        self.params = dict(cfg.params)

    def setup(self, ctx: SimContext) -> None:
        pass

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        return candidates

    def augment_selection(self, agent: Agent, task: np.ndarray, chosen: list[int], ctx: SimContext) -> list[int]:
        return chosen

    def observe(self, agent: Agent, peer: int, quality: float, ctx: SimContext) -> None:
        pass

    def audit(self, agent: Agent, peer: int, task: np.ndarray, peer_answer: int, ctx: SimContext) -> int:
        """Return the answer the agent finally uses (default: the peer's answer)."""
        return peer_answer

    def post_round(self, ctx: SimContext) -> None:
        pass


def make_defense(cfg: DefenseConfig) -> Defense:
    kind = cfg.kind
    if kind == "none":
        return Defense(cfg)
    if kind == "degree_bound":
        from sybagent.defenses.degree_bound import DegreeBound
        return DegreeBound(cfg)
    if kind == "random_mix":
        from sybagent.defenses.random_mix import RandomMix
        return RandomMix(cfg)
    if kind == "sybilrank":
        from sybagent.defenses.sybilrank import SybilRankDefense
        return SybilRankDefense(cfg)
    if kind == "sumup":
        from sybagent.defenses.sumup import SumUpDefense
        return SumUpDefense(cfg)
    if kind == "shared":
        from sybagent.defenses.shared import SharedDefense
        return SharedDefense(cfg)
    if kind == "proposed":
        from sybagent.defenses.proposed import ProposedDefense
        return ProposedDefense(cfg)
    raise ValueError(f"Unknown defense: {kind}")
