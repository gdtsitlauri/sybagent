"""Discovery registry and public review ledger."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from sybagent.agent import Agent


class Registry:
    """Answers capability queries by similarity between the query and advertised cards."""

    def __init__(self) -> None:
        self._agents: dict[int, Agent] = {}
        self._ids: list[int] = []
        self._cards: np.ndarray | None = None
        # adversarial bias on random sampling (models a compromised sampling service / DHT walk)
        self.bias_ids: list[int] = []
        self.bias_prob: float = 0.0
        self._cat_index: dict[int, list[int]] | None = None

    def register(self, agent: Agent) -> None:
        self._agents[agent.aid] = agent
        self._ids = list(self._agents)
        self._cards = None
        self._cat_index = None

    def unregister(self, aid: int) -> None:
        self._agents.pop(aid, None)
        self._ids = list(self._agents)
        self._cards = None
        self._cat_index = None

    def __contains__(self, aid: int) -> bool:
        return aid in self._agents

    def __len__(self) -> int:
        return len(self._agents)

    @property
    def ids(self) -> list[int]:
        return list(self._ids)

    def _card_matrix(self) -> np.ndarray:
        if self._cards is None:
            self._cards = np.stack([self._agents[i].card for i in self._ids])
        return self._cards

    def query(self, task: np.ndarray, m: int, exclude: int | None = None,
              override: dict[int, float] | None = None) -> list[int]:
        """Top-m agents by card/task cosine similarity (`override` replaces scores of query-adaptive identities)."""
        if not self._ids:
            return []
        scores = self._card_matrix() @ task
        if override:
            pos = {aid: i for i, aid in enumerate(self._ids)}
            for aid, sc in override.items():
                if aid in pos:
                    scores[pos[aid]] = sc
        order = np.argsort(-scores)
        out = []
        for idx in order:
            aid = self._ids[idx]
            if aid != exclude:
                out.append(aid)
            if len(out) == m:
                break
        return out

    def category_members(self, cat: int, always: set[int] | None = None) -> list[int]:
        """Agents whose advertised card's top skill is `cat`; `always` (query-adaptive identities) are in every category."""
        if self._cat_index is None:
            idx: dict[int, list[int]] = {}
            for aid in self._ids:
                idx.setdefault(int(np.argmax(self._agents[aid].card)), []).append(aid)
            self._cat_index = idx
        members = list(self._cat_index.get(cat, []))
        if always:
            have = set(members)
            members += [a for a in sorted(always) if a not in have and a in self._agents]
        return members

    def sample_category(self, cat: int, m: int, rng: np.random.Generator, exclude: int | None = None,
                        always: set[int] | None = None) -> list[int]:
        return self.sample(m, rng, exclude=exclude, pool=self.category_members(cat, always))

    def sample(self, m: int, rng: np.random.Generator, exclude: int | None = None,
               pool: list[int] | None = None) -> list[int]:
        """Uniform random sample of registered agents (or of `pool`)."""
        pool = [i for i in (self._ids if pool is None else pool) if i != exclude]
        if not pool:
            return []
        k = min(m, len(pool))
        out = [int(x) for x in rng.choice(pool, size=k, replace=False)]
        if self.bias_prob > 0 and self.bias_ids:
            biased = [b for b in self.bias_ids if b != exclude]
            for i in range(len(out)):
                if biased and rng.random() < self.bias_prob:
                    out[i] = int(biased[int(rng.integers(len(biased)))])
            out = list(dict.fromkeys(out))
        return out


class ReviewLedger:
    """Public reviews (rater -> ratee, score in [0, 1]), as in ERC-8004-style reputation registries."""

    def __init__(self) -> None:
        self._scores: dict[int, list[float]] = defaultdict(list)
        self.edges: list[tuple[int, int, float]] = []

    def add(self, rater: int, ratee: int, score: float) -> None:
        self._scores[ratee].append(score)
        self.edges.append((rater, ratee, score))

    def mean(self, aid: int, prior: float = 0.5, prior_weight: float = 1.0) -> float:
        s = self._scores.get(aid, [])
        return (prior * prior_weight + sum(s)) / (prior_weight + len(s))

    def count(self, aid: int) -> int:
        return len(self._scores.get(aid, []))
