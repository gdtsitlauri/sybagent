"""Round-based simulation loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sybagent.adversary import Adversary
from sybagent.agent import Agent
from sybagent.config import SimConfig
from sybagent.context import SimContext
from sybagent.defenses import make_defense
from sybagent.metrics import round_metrics
from sybagent.population import make_honest, sample_task
from sybagent.selection import make_selector
from sybagent.strategy import StrategicAttacker


class Simulation:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg
        self.ctx = SimContext(cfg=cfg, rng=np.random.default_rng(cfg.seed))
        self.selector = make_selector(cfg.selection)
        self.defense = make_defense(cfg.defense)
        self.adversary = Adversary(cfg.adversary, cfg.network)
        self.rows: list[dict] = []
        self.strategy: StrategicAttacker | None = None
        if cfg.adversary.enabled and cfg.adversary.policy != "fixed":
            self.strategy = StrategicAttacker(cfg.adversary, cfg.defense.params, cfg.rounds,
                                              np.random.default_rng([cfg.seed, 2]))
            self.ctx.answer_hook = self.strategy.answer
        self._init_population()

    def _init_population(self) -> None:
        ctx = self.ctx
        for a in make_honest(self.cfg.network.n_honest, self.cfg.network, ctx.rng):
            ctx.agents[a.aid] = a
            ctx.registry.register(a)
        if self.cfg.adversary.enabled:
            self.adversary.choose_targets([ctx.agents[i] for i in ctx.honest_ids], ctx.rng)
        self.defense.setup(ctx)

    def _spawn_sybils(self) -> None:
        ctx = self.ctx
        honest = [ctx.agents[i] for i in ctx.honest_ids]
        start = max(ctx.agents) + 1
        for s in self.adversary.spawn(honest, start, ctx.rng):
            ctx.agents[s.aid] = s
            ctx.registry.register(s)
        ctx.sybil_ids = self.adversary.sybil_ids
        ctx.registry.bias_ids = sorted(ctx.sybil_ids)
        ctx.registry.bias_prob = self.cfg.adversary.registry_bias
        if self.cfg.adversary.adaptive_cards:
            ctx.adaptive_ids = set(ctx.sybil_ids)
            ctx.adaptive_noise = self.cfg.adversary.adaptive_noise
        if self.cfg.network.real_cards:
            from sybagent.population import load_bundle
            b = load_bundle(self.cfg.network.real_cards)
            ctx.task_text = {t.tobytes(): str(x) for t, x in zip(b["task_emb"], b["task_text"])}
            if self.cfg.adversary.adaptive_cards:
                ctx.adaptive_lookup = {t.tobytes(): a for t, a in zip(b["task_emb"], b["adaptive_emb"])}
                ctx.adaptive_text = {t.tobytes(): str(x) for t, x in zip(b["task_emb"], b["adaptive_text"])}

    def step(self) -> dict:
        ctx, ncfg, acfg = self.ctx, self.cfg.network, self.cfg.adversary
        if acfg.enabled and ctx.round == acfg.join_round:
            self._spawn_sybils()
        if acfg.enabled and ctx.round == acfg.switch_round:
            for s in self.adversary.sybils:
                s.service_quality = acfg.post_switch_quality

        qualities: list[float] = []
        wrong_served = 0
        wrong_used = 0
        flips_bad = 0
        flips_good = 0
        sybil_wrong_served = 0
        sybil_wrong_used = 0
        n_answers = 0
        order = ctx.rng.permutation(ctx.honest_ids)
        for aid in order:
            agent = ctx.agents[int(aid)]
            for _ in range(self.cfg.tasks_per_agent_per_round):
                task = sample_task(agent, ncfg.task_noise, ctx.rng)
                found = ctx.registry.query(task, ncfg.candidates_per_query, exclude=agent.aid,
                                           override=ctx.adaptive_query_scores(task))
                candidates = list(dict.fromkeys(agent.neighbors + found))
                candidates = self.defense.filter_candidates(agent, task, candidates, ctx)
                chosen = self.selector.select(agent, task, candidates, ncfg.topk_select, ctx)
                chosen = self.defense.augment_selection(agent, task, chosen, ctx)
                correct = Agent.correct_answer(task)
                for peer in chosen:
                    q = ctx.agents[peer].quality_for(task, ctx.rng)
                    agent.remember(peer, q)
                    ctx.ledger.add(agent.aid, peer, q)
                    self.defense.observe(agent, peer, q, ctx)
                    qualities.append(q)
                    role = "serve_anchor" if peer in agent.pinned else "serve"
                    peer_ans = ctx.answer_of(peer, agent.aid, task, role)
                    n_answers += 1
                    is_sybil = peer in ctx.sybil_ids
                    if peer_ans != correct:
                        wrong_served += 1
                        if is_sybil:
                            sybil_wrong_served += 1
                    used = self.defense.audit(agent, peer, task, peer_ans, ctx)
                    wrong_used += int(used != correct)
                    if is_sybil:
                        sybil_wrong_used += int(used != correct)
                    if self.strategy is not None:
                        self.strategy.on_outcome(peer if is_sybil else None, agent.aid, peer_ans, used, correct)
                    if used != peer_ans:
                        if peer_ans == correct:
                            flips_bad += 1      # a correct answer was "corrected" into a wrong one
                        elif used == correct:
                            flips_good += 1
            agent.prune(ncfg.prune_below, ncfg.prune_min_obs)

        for rater, ratee, score in self.adversary.collusive_reviews(ctx.rng):
            ctx.ledger.add(rater, ratee, score)

        self.defense.post_round(ctx)
        row = round_metrics(ctx, self.adversary.targets, qualities)
        row["wrong_answer_rate"] = wrong_served / n_answers if n_answers else float("nan")
        row["wrong_used_rate"] = wrong_used / n_answers if n_answers else float("nan")
        row["bad_corrections"] = flips_bad
        row["good_corrections"] = flips_good
        row["sybil_wrong_served"] = sybil_wrong_served
        blocked = getattr(self.defense, "_blocked", {})
        row["blocked_sybil_pairs"] = sum(len(b & ctx.sybil_ids) for b in blocked.values())
        row["blocked_honest_pairs"] = sum(len(b - ctx.sybil_ids) for b in blocked.values())
        if hasattr(self.defense, "extra_metrics"):
            row.update(self.defense.extra_metrics(ctx))
            row["sybil_wrong_used"] = sybil_wrong_used
        if self.strategy is not None:
            row["sybil_wrong_used"] = sybil_wrong_used
            row["strategy_lie_frac"] = self.strategy.lies_told / max(1, self.strategy.decisions)
            self.strategy.end_round(blocked)
        self.rows.append(row)
        ctx.round += 1
        return row

    def run(self) -> pd.DataFrame:
        for _ in range(self.cfg.rounds):
            self.step()
        if self.strategy is not None:
            self.strategy.finish()
        return self.results()

    def results(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results().to_csv(path, index=False)
        return path
