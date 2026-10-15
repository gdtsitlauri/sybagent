"""Semantic-blind anchors.

Each honest agent splits its working set into
  * semantic slots, filled by the normal (card-driven) selector, and
  * `n_anchors` anchor slots, sampled uniformly from the registry WITHOUT reading cards,
    pinned against recency eviction and replaced only after observed poor service
    (exponentially weighted recent outcomes, so that Sybils that turn bad are evicted promptly).

Advertised cards cannot influence anchor membership, so the number of honest anchors does not depend on
how attractive the adversary's cards are, only on its share of identities.
"""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent
from sybagent.context import SimContext
from sybagent.defenses.base import Defense


class ProposedDefense(Defense):
    name = "proposed"

    def setup(self, ctx: SimContext) -> None:
        p = self.params
        self.n_anchors = int(p.get("n_anchors", 3))
        self.anchor_use = float(p.get("anchor_use", 0.5))
        self.evict_below = float(p.get("evict_below", 0.3))
        self.evict_min_obs = int(p.get("evict_min_obs", 2))
        self.churn_every = int(p.get("churn_every", 0))
        self.ewma = float(p.get("ewma", 0.3))   # weight of the newest outcome in the eviction statistic
        self._recent: dict[tuple[int, int], tuple[float, int]] = {}
        # answer cross-checking ("second opinion")
        self.audit_prob = float(p.get("audit_prob", 0.0))   # 0 disables cross-checking
        self.audit_flag = int(p.get("audit_flag", 3))       # disagreements with the anchor majority before blocking
        self.audit_decay = float(p.get("audit_decay", 0.0)) # per-round decay of disagreement counts
        self.audit_correct = bool(p.get("audit_correct", True))  # use the anchors' majority on disagreement
        self.audit_rule = str(p.get("audit_rule", "majority"))   # majority | unanimous
        self.audit_anchors = bool(p.get("audit_anchors", False))  # also cross-check anchors against the others
        # fraction of anchors drawn from the agent's own task category (by advertised top skill) instead of globally
        self.category_frac = float(p.get("anchor_category_frac", 0.0))
        self.n_cat = int(round(self.category_frac * self.n_anchors))
        self._kind: dict[tuple[int, int], str] = {}
        self._disagree: dict[tuple[int, int], float] = {}
        self._blocked: dict[int, set[int]] = {}
        # signed transcripts of disagreements: (agent, peer) -> [(task, the peer's answer)] (used by sharing)
        self._transcripts: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
        self._audited: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}
        base = ctx.cfg.network.max_neighbors
        for aid in ctx.honest_ids:
            ctx.agents[aid].max_neighbors = max(1, base - self.n_anchors)

    def _top_up(self, agent: Agent, ctx: SimContext) -> None:
        if self.n_cat == 0:
            self._fill(agent, ctx, self.n_anchors - len(agent.pinned), None)
            return
        n_cat_now = sum(self._kind.get((agent.aid, a)) == "cat" for a in agent.pinned)
        cat = int(np.argmax(agent.profile if agent.profile is not None else agent.capability))
        self._fill(agent, ctx, self.n_cat - n_cat_now, cat)
        self._fill(agent, ctx, self.n_anchors - len(agent.pinned), None)

    def _fill(self, agent: Agent, ctx: SimContext, need: int, cat: int | None) -> None:
        tries = 0
        while need > 0 and tries < 5:
            if cat is None:
                pool = ctx.registry.sample(need * 2 + 2, ctx.rng, exclude=agent.aid)
            else:
                pool = ctx.registry.sample_category(cat, need * 2 + 2, ctx.rng, exclude=agent.aid,
                                                    always=ctx.adaptive_ids)
            for peer in pool:
                if peer not in agent.pinned:
                    agent.pin(peer)
                    self._kind[(agent.aid, peer)] = "cat" if cat is not None else "glob"
                    need -= 1
                    if need == 0:
                        break
            tries += 1

    def filter_candidates(self, agent: Agent, task: np.ndarray, candidates: list[int], ctx: SimContext) -> list[int]:
        blocked = self._blocked.get(agent.aid)
        if not blocked:
            return candidates
        kept = [c for c in candidates if c not in blocked]
        return kept or candidates

    def augment_selection(self, agent: Agent, task: np.ndarray, chosen: list[int], ctx: SimContext) -> list[int]:
        self._top_up(agent, ctx)
        if not agent.pinned or ctx.rng.random() >= self.anchor_use:
            return chosen
        anchors = list(agent.pinned)
        a = anchors[int(ctx.rng.integers(len(anchors)))]
        if a in chosen:
            return chosen
        if len(chosen) >= ctx.cfg.network.topk_select and chosen:
            return chosen[:-1] + [a]
        return chosen + [a]

    def observe(self, agent: Agent, peer: int, quality: float, ctx: SimContext) -> None:
        if peer not in agent.pinned:
            return
        key = (agent.aid, peer)
        prev, n = self._recent.get(key, (quality, 0))
        val = quality if n == 0 else (1.0 - self.ewma) * prev + self.ewma * quality
        self._recent[key] = (val, n + 1)
        if n + 1 >= self.evict_min_obs and val < self.evict_below:
            agent.unpin(peer)
            self._recent.pop(key, None)

    def audit(self, agent: Agent, peer: int, task: np.ndarray, peer_answer: int, ctx: SimContext) -> int:
        """Second opinion: with prob. audit_prob, ask every anchor (other than `peer`) the same task and compare
        `peer`'s answer with the anchors' unique majority. On disagreement the agent uses the majority answer
        (correction); repeated disagreement blocks `peer` for this agent. Returns the answer the agent uses."""
        if self.audit_prob <= 0 or (peer in agent.pinned and not self.audit_anchors) \
                or ctx.answer_rng.random() >= self.audit_prob:
            return peer_answer
        refs = [a for a in agent.pinned if a != peer]
        if not refs:
            return peer_answer
        votes: dict[int, int] = {}
        for a in refs:
            ans = ctx.answer_of(a, agent.aid, task, "audit")
            votes[ans] = votes.get(ans, 0) + 1
        ctx.audit_calls += len(refs)
        top = max(votes.values())
        majority = [ans for ans, c in votes.items() if c == top]
        if len(majority) != 1:          # tie among anchors: no evidence either way
            return peer_answer
        if self.audit_rule == "unanimous" and (len(votes) != 1 or len(refs) < 2):
            return peer_answer          # act only if every (>= 2) anchor gives the same answer
        # every conclusive audit is kept as a signed transcript item (agreements too), for shared defenses
        self._audited.setdefault((agent.aid, peer), []).append((task, peer_answer))
        if peer_answer == majority[0]:
            return peer_answer
        key = (agent.aid, peer)
        self._disagree[key] = self._disagree.get(key, 0.0) + 1.0
        self._transcripts.setdefault(key, []).append((task, peer_answer))
        if self._disagree[key] >= self.audit_flag:
            self._blocked.setdefault(agent.aid, set()).add(peer)
            agent.forget(peer)
            if peer in agent.pinned:
                agent.unpin(peer)
                self._recent.pop(key, None)
        return majority[0] if self.audit_correct else peer_answer

    def post_round(self, ctx: SimContext) -> None:
        if self.audit_decay > 0 and self._disagree:
            for k in list(self._disagree):
                self._disagree[k] *= 1.0 - self.audit_decay
        if not self.churn_every or ctx.round % self.churn_every:
            return
        for aid in ctx.honest_ids:
            agent = ctx.agents[aid]
            if agent.pinned:
                victim = list(agent.pinned)[int(ctx.rng.integers(len(agent.pinned)))]
                agent.unpin(victim)
                self._recent.pop((aid, victim), None)


def anchor_eclipse_bound(n_honest: int, n_sybils: int, n_anchors: int, beta: float = 0.0) -> float:
    """Theorem 1: P(no honest anchor) <= p^a with p = beta + (1 - beta) * B / (N + B - a)."""
    pool = n_honest + n_sybils - n_anchors
    if pool <= 0:
        return 1.0
    p = beta + (1.0 - beta) * n_sybils / pool
    return min(1.0, p) ** n_anchors
