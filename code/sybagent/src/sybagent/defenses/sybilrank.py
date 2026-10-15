"""SybilRank (Cao et al., NSDI 2012) on the review graph: early-terminated power iteration from trusted seeds."""

from __future__ import annotations

import math

import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.base import Defense


def sybilrank_scores(adj: dict[int, list[int]], seeds: list[int], iterations: int | None = None) -> dict[int, float]:
    nodes = list(adj)
    if not nodes:
        return {}
    trust = {v: 0.0 for v in nodes}
    live_seeds = [s for s in seeds if s in trust]
    for s in live_seeds:
        trust[s] = 1.0 / len(live_seeds)
    iters = iterations or max(1, math.ceil(math.log2(len(nodes))))
    for _ in range(iters):
        nxt = {v: 0.0 for v in nodes}
        for u in nodes:
            d = len(adj[u])
            if d == 0 or trust[u] == 0.0:
                continue
            share = trust[u] / d
            for v in adj[u]:
                nxt[v] += share
        trust = nxt
    return {v: (trust[v] / len(adj[v]) if adj[v] else 0.0) for v in nodes}


class SybilRankDefense(Defense):
    name = "sybilrank"

    def setup(self, ctx: SimContext) -> None:
        self.n_seeds = int(self.params.get("n_seeds", 10))
        self.drop_quantile = float(self.params.get("drop_quantile", 0.2))
        self.every = int(self.params.get("every", 5))
        honest = ctx.honest_ids
        k = min(self.n_seeds, len(honest))
        self.seeds = [int(x) for x in ctx.rng.choice(honest, size=k, replace=False)] if k else []
        self._score: dict[int, float] = {}
        self._cut = -math.inf

    def post_round(self, ctx: SimContext) -> None:
        if ctx.round % self.every:
            return
        g = ctx.review_graph()
        adj = {v: list(g.neighbors(v)) for v in g.nodes}
        self._score = sybilrank_scores(adj, self.seeds)
        vals = np.array(list(self._score.values()))
        self._cut = float(np.quantile(vals, self.drop_quantile)) if len(vals) else -math.inf

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        if not self._score:
            return candidates
        kept = [c for c in candidates if self._score.get(c, 0.0) > self._cut]
        return kept or candidates
