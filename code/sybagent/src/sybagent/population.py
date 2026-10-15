"""Creation of honest agents and task sampling."""

from __future__ import annotations

import numpy as np

from sybagent.agent import Agent, unit
from sybagent.cards import render_card
from sybagent.config import NetworkConfig


def random_unit(dim: int, rng: np.random.Generator) -> np.ndarray:
    return unit(np.abs(rng.normal(size=dim)))


_BUNDLES: dict[str, dict] = {}


def load_bundle(path: str) -> dict:
    if path not in _BUNDLES:
        with np.load(path, allow_pickle=False) as z:
            _BUNDLES[path] = {k: z[k] for k in z.files}
    b = _BUNDLES[path]
    if "correct" in b:                                   # real-answer bundle: register the real answers
        from sybagent import agent as agent_mod
        agent_mod.REAL_QA.clear()
        for t, s, c, e, g in zip(b["task_emb"], b["task_subject"], b["correct"], b["expert"], b["generic"]):
            agent_mod.REAL_QA[t.tobytes()] = (int(s), int(c), int(e), int(g))
    return b


def make_honest(n: int, cfg: NetworkConfig, rng: np.random.Generator, start_id: int = 0) -> list[Agent]:
    if cfg.real_cards:
        return _make_honest_real(n, cfg, start_id)
    agents = []
    for k in range(n):
        aid = start_id + k
        cap = random_unit(cfg.capability_dim, rng)
        card = unit(cap + rng.normal(0.0, cfg.capability_noise, size=cfg.capability_dim))
        agents.append(
            Agent(
                aid=aid,
                honest=True,
                capability=cap,
                card=card,
                profile=random_unit(cfg.capability_dim, rng),
                card_text=render_card(aid, card),
                max_neighbors=cfg.max_neighbors,
                answer_correct_prob=cfg.honest_answer_correct,
                answer_model=cfg.answer_model,
                skill_answer_lo=cfg.skill_answer_lo,
                skill_answer_hi=cfg.skill_answer_hi,
            )
        )
    return agents


def _make_honest_real(n: int, cfg: NetworkConfig, start_id: int) -> list[Agent]:
    b = load_bundle(cfg.real_cards)
    if n > len(b["honest_emb"]):
        raise ValueError(f"bundle has only {len(b['honest_emb'])} honest cards")
    out = []
    for k in range(n):
        tasks = b["task_emb"][b["agent_tasks"][k]]
        out.append(Agent(aid=start_id + k, honest=True, capability=b["honest_emb"][k], card=b["honest_emb"][k],
                         profile=tasks[0], card_text=f"Agent {start_id + k}: {b['honest_text'][k]}",
                         max_neighbors=cfg.max_neighbors, answer_correct_prob=cfg.honest_answer_correct,
                         answer_model=cfg.answer_model, skill_answer_lo=cfg.skill_answer_lo,
                         skill_answer_hi=cfg.skill_answer_hi, tasks=tasks,
                         specialty=int(b["specialty"][k]) if "specialty" in b else -1))
    return out


def sample_task(agent: Agent, noise: float, rng: np.random.Generator) -> np.ndarray:
    if agent.tasks is not None:
        return agent.tasks[int(rng.integers(len(agent.tasks)))]
    base = agent.profile if agent.profile is not None else agent.capability
    return unit(np.abs(base + rng.normal(0.0, noise, size=base.shape)))
