"""Vector-based selectors: a fast stand-in for semantic (LLM) matching at large scale."""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.selection.base import Selector


def _softmax_sample(scores: np.ndarray, k: int, temperature: float, rng: np.random.Generator) -> np.ndarray:
    k = min(k, len(scores))
    if temperature <= 0:
        return np.argsort(-scores)[:k]
    z = (scores - scores.max()) / temperature
    p = np.exp(z)
    p /= p.sum()
    return rng.choice(len(scores), size=k, replace=False, p=p)


class SimilaritySelector(Selector):
    def scores(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> np.ndarray:
        s = ctx.card_scores(candidates, task)
        w = self.cfg.experience_weight
        if w:
            exp = np.array([agent.known_quality(c) - 0.5 for c in candidates])
            s = s + w * exp
        if ctx.selection_bonus:
            s = s + np.array([ctx.selection_bonus.get(c, 0.0) for c in candidates])
        return s

    def select(self, agent: Agent, task: np.ndarray, candidates: list[int], k: int, ctx: SimContext) -> list[int]:
        if not candidates:
            return []
        s = self.scores(agent, task, candidates, ctx)
        idx = _softmax_sample(s, k, self.cfg.temperature, ctx.rng)
        return [candidates[i] for i in idx]


class ReputationSelector(SimilaritySelector):
    """Similarity blended with the public review mean (collusive reviews inflate this)."""

    def scores(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> np.ndarray:
        sim = super().scores(agent, task, candidates, ctx)
        rep = np.array([ctx.ledger.mean(c) for c in candidates])
        w = self.cfg.reputation_weight
        return (1.0 - w) * sim + w * rep
