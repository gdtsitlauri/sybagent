"""In-degree bound (Singh et al., eclipse defense for overlays): skip peers that are far more popular than average."""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.base import Defense


class DegreeBound(Defense):
    name = "degree_bound"

    def setup(self, ctx: SimContext) -> None:
        self.factor = float(self.params.get("factor", 2.0))
        self._deg: dict[int, int] = {}
        self._cap = float("inf")

    def post_round(self, ctx: SimContext) -> None:
        self._deg = ctx.in_degrees()
        vals = np.array(list(self._deg.values()), dtype=float)
        mean = vals.mean() if len(vals) else 0.0
        self._cap = max(1.0, self.factor * mean)

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        kept = [c for c in candidates if self._deg.get(c, 0) <= self._cap]
        return kept or candidates
