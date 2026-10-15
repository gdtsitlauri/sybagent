"""Shared simulation state passed to selectors and defenses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import networkx as nx
import numpy as np

from sybagent.agent import Agent
from sybagent.config import SimConfig
from sybagent.registry import Registry, ReviewLedger


@dataclass
class SimContext:
    cfg: SimConfig
    rng: np.random.Generator
    agents: dict[int, Agent] = field(default_factory=dict)
    registry: Registry = field(default_factory=Registry)
    ledger: ReviewLedger = field(default_factory=ReviewLedger)
    round: int = 0
    sybil_ids: set[int] = field(default_factory=set)  # ground truth, for metrics only
    llm_calls: int = 0
    audit_calls: int = 0            # extra anchor queries spent on cross-checking
    # separate stream for task answers and audits, so that enabling them leaves selection randomness unchanged
    answer_rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.answer_rng is None:
            self.answer_rng = np.random.default_rng([self.cfg.seed, 1])
    adaptive_ids: set[int] = field(default_factory=set)  # identities that tailor cards per query
    adaptive_noise: float = 0.05
    # strategic adversary: (sybil, asker, task, role) -> answer; role is "serve", "serve_anchor" or "audit"
    answer_hook: Callable[[int, int, np.ndarray, str], int] | None = None

    def answer_of(self, peer: int, asker: int, task: np.ndarray, role: str) -> int:
        """Answer of `peer` to `asker`; Sybils follow the strategic policy when one is installed."""
        if self.answer_hook is not None and peer in self.sybil_ids:
            return self.answer_hook(peer, asker, task, role)
        return self.agents[peer].answer(task, self.answer_rng)

    # verified reputation (study 22): peer -> score bonus added by card-based selectors; empty = unchanged
    selection_bonus: dict = field(default_factory=dict)
    verify_calls: int = 0

    # real-card mode: task bytes -> LLM-written adaptive card (embedding and text), task text
    adaptive_lookup: dict = field(default_factory=dict)
    adaptive_text: dict = field(default_factory=dict)
    task_text: dict = field(default_factory=dict)

    def _adaptive_vec(self, task: np.ndarray, size: int) -> np.ndarray:
        base = self.adaptive_lookup[task.tobytes()]
        v = base[None, :] + self.rng.normal(0.0, self.adaptive_noise, size=(size, task.shape[0]))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def task_label(self, task: np.ndarray) -> str | None:
        return self.task_text.get(task.tobytes())

    def card_for(self, aid: int, task: np.ndarray) -> np.ndarray:
        """Card as seen when evaluated for `task` (query-adaptive for adaptive identities)."""
        if aid in self.adaptive_ids and self.adaptive_lookup:
            return self._adaptive_vec(task, 1)[0]
        if aid in self.adaptive_ids:
            v = np.abs(task + self.rng.normal(0.0, self.adaptive_noise, size=task.shape))
            return v / np.linalg.norm(v)
        return self.agents[aid].card

    def adaptive_query_scores(self, task: np.ndarray) -> dict[int, float]:
        """Vectorized scores of all query-adaptive identities for one discovery query."""
        if not self.adaptive_ids:
            return {}
        ids = sorted(self.adaptive_ids)
        if self.adaptive_lookup:
            return dict(zip(ids, (self._adaptive_vec(task, len(ids)) @ task).tolist()))
        v = np.abs(task[None, :] + self.rng.normal(0.0, self.adaptive_noise, size=(len(ids), task.shape[0])))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return dict(zip(ids, (v @ task).tolist()))

    def card_scores(self, ids: list[int], task: np.ndarray) -> np.ndarray:
        return np.array([float(self.card_for(i, task) @ task) for i in ids])

    @property
    def honest_ids(self) -> list[int]:
        return [a.aid for a in self.agents.values() if a.honest]

    def peer_graph(self) -> nx.DiGraph:
        """Directed graph: edge u -> v if v is in u's working set."""
        g = nx.DiGraph()
        g.add_nodes_from(self.agents)
        for a in self.agents.values():
            for p in a.neighbors:
                g.add_edge(a.aid, p)
        return g

    def review_graph(self) -> nx.Graph:
        """Undirected graph of review interactions (who rated whom)."""
        g = nx.Graph()
        g.add_nodes_from(self.agents)
        for rater, ratee, _ in self.ledger.edges:
            if rater != ratee:
                g.add_edge(rater, ratee)
        return g

    def in_degrees(self) -> dict[int, int]:
        deg = {aid: 0 for aid in self.agents}
        for a in self.agents.values():
            for p in a.neighbors:
                if p in deg:
                    deg[p] += 1
        return deg
