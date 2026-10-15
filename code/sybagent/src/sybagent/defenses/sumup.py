"""SumUp-style vote filtering (Tran et al., NSDI 2009): only reviews that fit through a max-flow to a trusted collector count."""

from __future__ import annotations

from collections import defaultdict

import networkx as nx
import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.base import Defense

SUPER_SOURCE = -1


def accepted_voters(g: nx.Graph, collector: int, voters: set[int], near_capacity: int = 4) -> set[int]:
    """Voters whose unit vote reaches `collector` in a max-flow; edges next to the collector get extra capacity."""
    if collector not in g or not voters:
        return set()
    near = set(g.neighbors(collector)) | {collector}
    flow_net = nx.DiGraph()
    for u, v in g.edges:
        cap = near_capacity if (u in near and v in near) else 1
        flow_net.add_edge(u, v, capacity=cap)
        flow_net.add_edge(v, u, capacity=cap)
    for v in voters:
        if v in g and v != collector:
            flow_net.add_edge(SUPER_SOURCE, v, capacity=1)
    if SUPER_SOURCE not in flow_net:
        return set()
    _, flow = nx.maximum_flow(flow_net, SUPER_SOURCE, collector)
    return {v for v, f in flow[SUPER_SOURCE].items() if f > 0}


class SumUpDefense(Defense):
    name = "sumup"

    def setup(self, ctx: SimContext) -> None:
        self.every = int(self.params.get("every", 5))
        self.threshold = float(self.params.get("threshold", 0.4))
        self.near_capacity = int(self.params.get("near_capacity", 4))
        honest = ctx.honest_ids
        self.collector = int(ctx.rng.choice(honest)) if honest else None
        self._rep: dict[int, float] = {}

    def post_round(self, ctx: SimContext) -> None:
        if ctx.round % self.every or self.collector is None:
            return
        g = ctx.review_graph()
        voters = {r for r, _, _ in ctx.ledger.edges}
        ok = accepted_voters(g, self.collector, voters, self.near_capacity)
        sums: dict[int, list[float]] = defaultdict(list)
        for rater, ratee, score in ctx.ledger.edges:
            if rater in ok:
                sums[ratee].append(score)
        self._rep = {k: float(np.mean(v)) for k, v in sums.items()}

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        if not self._rep:
            return candidates
        kept = [c for c in candidates if self._rep.get(c, 1.0) >= self.threshold]
        return kept or candidates
