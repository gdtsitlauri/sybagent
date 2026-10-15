"""Agent model: true capability, advertised card, working set of peers."""

from __future__ import annotations

import zlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np


SKILL_MATCH_FLOOR = 0.6

# real-answer mode (study 17): task bytes -> (subject, correct option, expert answer, non-specialist answer); the
# answers are a real LLM's, computed once (analysis/build_real_answers.py); -1 means the reply was unparsable
REAL_QA: dict[bytes, tuple[int, int, int, int]] = {}


def unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


@dataclass
class Agent:
    aid: int
    honest: bool
    capability: np.ndarray          # true skill direction (unit vector)
    card: np.ndarray                # advertised skill direction (unit vector)
    profile: np.ndarray | None = None  # direction around which this agent's own tasks are drawn
    card_text: str = ""
    max_neighbors: int = 8          # capacity of the recency-managed part of the working set
    service_quality: float = 0.6    # used only when honest is False
    answer_correct_prob: float = 0.95  # probability of returning the correct task answer
    answer_model: str = "flat"      # honest agents only: flat | skill
    skill_answer_lo: float = 0.5
    skill_answer_hi: float = 0.98
    coordinated_lies: bool = False  # Sybils only: every lie on a task is the same wrong answer
    tasks: np.ndarray | None = None  # real-card mode: this agent's task embeddings (rows)
    specialty: int = -1             # real-answer mode: the subject this honest agent is an expert in
    # peer id -> running mean of observed outcome quality, ordered by recency
    peers: OrderedDict[int, float] = field(default_factory=OrderedDict)
    peer_counts: dict[int, int] = field(default_factory=dict)
    # peers pinned by a defense: never evicted by recency, never pruned by the agent itself
    pinned: dict[int, float] = field(default_factory=dict)
    pinned_counts: dict[int, int] = field(default_factory=dict)

    def quality_for(self, task: np.ndarray, rng: np.random.Generator) -> float:
        """Outcome quality in [0, 1] when this agent executes `task`."""
        if not self.honest:
            return float(np.clip(rng.normal(self.service_quality, 0.05), 0.0, 1.0))
        match = float(np.dot(self.capability, task))
        return float(np.clip(0.5 * (match + 1.0) + rng.normal(0.0, 0.05), 0.0, 1.0))

    def answer(self, task: np.ndarray, rng: np.random.Generator, n_classes: int = 8) -> int:
        """Return a task 'answer'. The correct answer is a deterministic function of the task."""
        qa = REAL_QA.get(task.tobytes()) if REAL_QA else None
        if qa is not None:
            subject, correct, expert, generic = qa
            if self.honest:
                ans = expert if subject == self.specialty else generic
                return ans if ans >= 0 else (correct + 2) % 4    # an unparsable reply counts as wrong
            if rng.random() < self.answer_correct_prob:
                return correct
            if self.coordinated_lies:
                return (correct + 1) % 4
            wrong = int(rng.integers(3))
            return wrong if wrong < correct else wrong + 1
        correct = self.correct_answer(task, n_classes)
        p = self.answer_correct_prob
        if self.honest and self.answer_model == "skill":
            # random non-negative skill vectors already have cosine ~0.64, so rescale [floor, 1] -> [0, 1]
            cos = float(np.dot(self.capability, task))
            match = min(1.0, max(0.0, (cos - SKILL_MATCH_FLOOR) / (1.0 - SKILL_MATCH_FLOOR)))
            p = self.skill_answer_lo + (self.skill_answer_hi - self.skill_answer_lo) * match
        if rng.random() < p:
            return correct
        if self.coordinated_lies:
            return (correct + 1) % n_classes     # the adversary's agreed wrong answer for this task
        wrong = int(rng.integers(n_classes - 1))
        return wrong if wrong < correct else wrong + 1

    @staticmethod
    def correct_answer(task: np.ndarray, n_classes: int = 8) -> int:
        if REAL_QA:
            qa = REAL_QA.get(task.tobytes())
            if qa is not None:
                return qa[1]
        return int(zlib.crc32(np.ascontiguousarray(task).tobytes()) % n_classes)

    def remember(self, peer: int, quality: float) -> None:
        if peer in self.pinned:
            n = self.pinned_counts.get(peer, 0) + 1
            prev = self.pinned[peer]
            self.pinned[peer] = prev + (quality - prev) / n
            self.pinned_counts[peer] = n
            return
        n = self.peer_counts.get(peer, 0) + 1
        prev = self.peers.pop(peer, 0.0)
        self.peers[peer] = prev + (quality - prev) / n
        self.peer_counts[peer] = n
        while len(self.peers) > self.max_neighbors:
            evicted, _ = self.peers.popitem(last=False)
            self.peer_counts.pop(evicted, None)

    def forget(self, peer: int) -> None:
        self.peers.pop(peer, None)
        self.peer_counts.pop(peer, None)

    def pin(self, peer: int) -> None:
        q = self.peers.pop(peer, None)
        c = self.peer_counts.pop(peer, 0)
        self.pinned[peer] = q if q is not None else 0.0
        self.pinned_counts[peer] = c

    def unpin(self, peer: int) -> None:
        self.pinned.pop(peer, None)
        self.pinned_counts.pop(peer, None)

    def prune(self, below: float, min_obs: int) -> list[int]:
        dropped = [p for p, q in self.peers.items() if self.peer_counts.get(p, 0) >= min_obs and q < below]
        for p in dropped:
            self.forget(p)
        return dropped

    def known_quality(self, peer: int, default: float = 0.5) -> float:
        if peer in self.pinned and self.pinned_counts.get(peer, 0):
            return self.pinned[peer]
        return self.peers.get(peer, default)

    @property
    def neighbors(self) -> list[int]:
        return list(self.pinned.keys()) + list(self.peers.keys())
