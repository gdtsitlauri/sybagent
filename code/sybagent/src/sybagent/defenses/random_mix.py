"""Random-sampling mix (in the spirit of Brahms/Basalt): part of every candidate list is drawn uniformly at random."""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.base import Defense


class RandomMix(Defense):
    name = "random_mix"

    def setup(self, ctx: SimContext) -> None:
        self.fraction = float(self.params.get("fraction", 0.3))

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        n_rand = int(round(self.fraction * len(candidates)))
        if n_rand == 0:
            return candidates
        keep = candidates[: len(candidates) - n_rand]
        pool = ctx.registry.sample(n_rand * 3, ctx.rng, exclude=agent.aid)
        extra = [p for p in pool if p not in keep][:n_rand]
        return keep + extra
