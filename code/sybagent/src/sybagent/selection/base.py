"""Peer selection interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sybagent.agent import Agent
from sybagent.config import SelectionConfig
from sybagent.context import SimContext


class Selector(ABC):
    def __init__(self, cfg: SelectionConfig) -> None:
        self.cfg = cfg

    @abstractmethod
    def select(self, agent: Agent, task: np.ndarray, candidates: list[int], k: int, ctx: SimContext) -> list[int]:
        """Return up to k peer ids chosen from `candidates` for `task`."""


def make_selector(cfg: SelectionConfig) -> Selector:
    if cfg.kind == "similarity":
        from sybagent.selection.similarity import SimilaritySelector
        return SimilaritySelector(cfg)
    if cfg.kind == "reputation":
        from sybagent.selection.similarity import ReputationSelector
        return ReputationSelector(cfg)
    if cfg.kind == "llm":
        from sybagent.selection.llm import LLMSelector
        return LLMSelector(cfg)
    raise ValueError(f"Unknown selector: {cfg.kind}")
