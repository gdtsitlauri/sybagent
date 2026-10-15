"""Sybil adversary: cheap identities whose advertised cards are aligned with target agents' task profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sybagent.agent import Agent, unit
from sybagent.cards import render_card
from sybagent.config import AdversaryConfig, NetworkConfig
from sybagent.population import random_unit


@dataclass
class Adversary:
    cfg: AdversaryConfig
    net_cfg: NetworkConfig
    targets: list[int] = field(default_factory=list)
    sybils: list[Agent] = field(default_factory=list)

    def choose_targets(self, honest: list[Agent], rng: np.random.Generator) -> list[int]:
        k = min(self.cfg.n_targets, len(honest))
        self.targets = [int(a) for a in rng.choice([h.aid for h in honest], size=k, replace=False)]
        return self.targets

    def spawn(self, honest: list[Agent], start_id: int, rng: np.random.Generator) -> list[Agent]:
        by_id = {h.aid: h for h in honest}
        if self.net_cfg.real_cards:
            return self._spawn_real(honest, start_id, rng)
        a = float(np.clip(self.cfg.attractiveness, 0.0, 1.0))
        out = []
        for k in range(self.cfg.n_sybils):
            aid = start_id + k
            if self.cfg.strategy == "targeted" and self.targets:
                aim = by_id[self.targets[k % len(self.targets)]].profile
            else:
                aim = honest[int(rng.integers(len(honest)))].profile
            own = random_unit(self.net_cfg.capability_dim, rng)
            card = unit((1.0 - a) * own + a * aim)
            if self.cfg.card_jitter > 0:
                card = unit(np.abs(card + rng.normal(0.0, self.cfg.card_jitter, size=card.shape)))
            out.append(
                Agent(
                    aid=aid,
                    honest=False,
                    capability=own,
                    card=card,
                    profile=own,
                    card_text=render_card(aid, card),
                    max_neighbors=self.net_cfg.max_neighbors,
                    service_quality=self.cfg.service_quality,
                    answer_correct_prob=1.0 - self.cfg.answer_wrong_prob,
                    coordinated_lies=self.cfg.coordinated_lies,
                )
            )
        self.sybils = out
        return out

    def _spawn_real(self, honest: list[Agent], start_id: int, rng: np.random.Generator) -> list[Agent]:
        from sybagent.population import load_bundle
        b = load_bundle(self.net_cfg.real_cards)
        out = []
        for k in range(self.cfg.n_sybils):
            if self.cfg.strategy == "targeted" and self.targets:
                t = self.targets[k % len(self.targets)]
            else:
                t = honest[int(rng.integers(len(honest)))].aid
            card = b["static_emb"][t] + rng.normal(0.0, self.cfg.card_jitter / 4, size=b["static_emb"].shape[1])
            card = unit(card)
            out.append(Agent(aid=start_id + k, honest=False, capability=card, card=card, profile=card,
                             card_text=f"Agent {start_id + k}: {b['static_text'][t]}",
                             max_neighbors=self.net_cfg.max_neighbors, service_quality=self.cfg.service_quality,
                             answer_correct_prob=1.0 - self.cfg.answer_wrong_prob,
                             coordinated_lies=self.cfg.coordinated_lies))
        self.sybils = out
        return out

    def collusive_reviews(self, rng: np.random.Generator) -> list[tuple[int, int, float]]:
        if not self.cfg.collusive_reviews or len(self.sybils) < 2:
            return []
        ids = [s.aid for s in self.sybils]
        reviews = []
        for s in ids:
            others = [x for x in ids if x != s]
            k = min(self.cfg.reviews_per_sybil, len(others))
            for t in rng.choice(others, size=k, replace=False):
                reviews.append((s, int(t), 1.0))
        return reviews

    @property
    def sybil_ids(self) -> set[int]:
        return {s.aid for s in self.sybils}
