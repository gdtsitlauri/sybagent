"""Strategic Sybil answer policies (study 11).

The adversary decides, per (Sybil, victim) interaction, whether to tell the truth or to lie. Lies are coordinated
(every lying Sybil gives the same wrong answer for a task), the strongest choice against majority second opinions.

Roles in which a Sybil is asked:
  serve         the victim picked it by its card; the answer may be cross-checked by the victim's anchors
  serve_anchor  the Sybil is one of the victim's anchors and serves a task
  audit         the Sybil is an anchor giving a second opinion on another peer's answer

Policies:
  budget  knowledge-based heuristic: knows the defense (Kerckhoffs) and lies to each victim at most
          `budget_lies` times in role serve (default audit_flag - 1, so it is never blocked), and always lies as
          anchor or auditor.
  rl      tabular Q-learning (Sutton & Barto, ch. 6) shared by all Sybils. State = (role, lies told to this victim
          so far (capped), phase of the run). Reward = 1 when the victim ends up USING a wrong answer because of the
          decision (a lie that was not corrected, or, as auditor, a correct answer "corrected" into a wrong one).
          Being blocked ends the (Sybil, victim) episode, so the learner trades lies now against future access.
          It is trained in the simulator with ground-truth rewards: an upper bound on a real attacker's knowledge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sybagent.agent import Agent

ROLES = ("serve", "serve_anchor", "audit")
MAX_LIES = 6
N_PHASES = 4


class StrategicAttacker:
    def __init__(self, cfg, defense_params: dict, rounds: int, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self.kind = cfg.policy
        self.rng = rng
        self.rounds = max(1, rounds)
        flag = int(defense_params.get("audit_flag", 3))
        self.budget = cfg.budget_lies if cfg.budget_lies >= 0 else max(0, flag - 1)
        self.lies: dict[tuple[int, int], int] = {}          # (sybil, victim) -> lies told in role serve
        self.round = 0
        # Q-learning
        self.q = np.zeros((len(ROLES), MAX_LIES + 1, N_PHASES, 2))
        if self.kind == "rl" and cfg.policy_q:
            self.q = np.array(json.loads(Path(cfg.policy_q).read_text())["q"])
        self.train = bool(cfg.policy_train) and self.kind == "rl"
        self._pending: dict[tuple[int, int], tuple[tuple[int, int, int], int, float]] = {}
        self._audit_buf: list[tuple[int, int, tuple[int, int, int], int]] = []
        self.decisions = 0
        self.lies_told = 0

    # ---- state ----
    def _state(self, sybil: int, victim: int, role: str) -> tuple[int, int, int]:
        lies = min(self.lies.get((sybil, victim), 0), MAX_LIES)
        phase = min(N_PHASES - 1, self.round * N_PHASES // self.rounds)
        return ROLES.index(role), lies, phase

    def _choose(self, s: tuple[int, int, int], sybil: int, victim: int, role: str) -> int:
        if self.kind == "sleeper":
            return int(self.round >= self.cfg.sleeper_round)
        if self.kind == "budget":
            if role == "serve":
                return int(self.lies.get((sybil, victim), 0) < self.budget)
            return 1
        if self.train and self.rng.random() < self.cfg.policy_epsilon:
            return int(self.rng.integers(2))
        qs = self.q[s]
        return int(qs[1] >= qs[0])          # unvisited states (ties) default to lying, the strongest fixed policy

    # ---- hooks called by the simulation ----
    def answer(self, sybil: int, victim: int, task: np.ndarray, role: str) -> int:
        s = self._state(sybil, victim, role)
        if self.train:
            self._bootstrap((sybil, victim), s)
        act = self._choose(s, sybil, victim, role)
        self.decisions += 1
        correct = Agent.correct_answer(task)
        if act:
            self.lies_told += 1
            if role == "serve":
                self.lies[(sybil, victim)] = self.lies.get((sybil, victim), 0) + 1
        if role == "audit":
            self._audit_buf.append((sybil, victim, s, act))
        elif self.train:
            self._pending[(sybil, victim)] = (s, act, 0.0)
        return (correct + 1) % 8 if act else correct

    def on_outcome(self, sybil: int | None, victim: int, peer_answer: int, used: int, correct: int) -> None:
        """After an answer was served (and possibly audited): assign rewards."""
        if self.train:
            if sybil is not None and (sybil, victim) in self._pending:
                s, a, _ = self._pending[(sybil, victim)]
                self._pending[(sybil, victim)] = (s, a, float(used != correct))
            bad_fix = float(used != correct and peer_answer == correct)
            for syb, vic, s, a in self._audit_buf:
                self._update(s, a, bad_fix if a else 0.0, None)
        self._audit_buf.clear()

    def end_round(self, blocked: dict[int, set[int]]) -> None:
        if self.train:
            for (syb, vic), (s, a, r) in list(self._pending.items()):
                if syb in blocked.get(vic, ()):
                    self._update(s, a, r, None)          # blocked: terminal
                    del self._pending[(syb, vic)]
        self.round += 1

    def finish(self) -> None:
        if self.train:
            for s, a, r in self._pending.values():
                self._update(s, a, r, None)
            self._pending.clear()

    # ---- learning ----
    def _bootstrap(self, key: tuple[int, int], s_next: tuple[int, int, int]) -> None:
        if key in self._pending:
            s, a, r = self._pending.pop(key)
            self._update(s, a, r, s_next)

    def _update(self, s, a: int, r: float, s_next) -> None:
        target = r if s_next is None else r + self.cfg.policy_gamma * float(self.q[s_next].max())
        self.q[s + (a,)] += self.cfg.policy_alpha * (target - self.q[s + (a,)])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"q": self.q.tolist(), "roles": ROLES, "max_lies": MAX_LIES,
                                          "phases": N_PHASES}))

    def greedy_policy(self) -> dict:
        """Readable summary: lie (1) or truth (0) per state."""
        out = {}
        for r, role in enumerate(ROLES):
            out[role] = [[int(self.q[r, l, p, 1] >= self.q[r, l, p, 0]) for p in range(N_PHASES)]
                         for l in range(MAX_LIES + 1)]
        return out
