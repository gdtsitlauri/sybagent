"""Exact best response of a Sybil by dynamic programming (study 11, Proposition 7).

One (Sybil, victim) slot over a horizon of T interactions. Each interaction the Sybil answers a task: truth or lie.
  * A lie is audited and caught with probability q = u * h (audit coin u, anchors give a unique correct majority h);
    a caught lie is corrected (no damage) and adds one disagreement. An uncaught lie is used: damage 1.
  * A truthful answer is (wrongly) flagged with probability f (colluding anchors; f = 0 by default).
  * At kappa disagreements the Sybil is blocked for this victim.
The Sybil never sees the audit coin. What it knows is how many lies and truths the current identity has told and
that it is still served. The number of disagreements d is then distributed as a sum of two binomials conditioned on
d < kappa, so (t, age of the identity, lies of the identity) is an exact state: a finite MDP solved by backward
induction (value iteration over the horizon).

When the Sybil is blocked:
  * replaceable  (enough identities, local blocking): a fresh Sybil takes the slot (age 0);
  * scarce       (one identity per slot, or blocking shared network-wide): the slot is lost.

    python analysis/dp_attacker.py
"""

from __future__ import annotations

from functools import lru_cache
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "11_strategic_attacker" / "dp"


def d_dist(lies: int, truths: int, q: float, f: float, kappa: int) -> np.ndarray:
    """P(d = 0..kappa-1) (unnormalized: the rest is the probability of having been blocked)."""
    a = np.array([comb(lies, i) * q ** i * (1 - q) ** (lies - i) for i in range(min(lies, kappa - 1) + 1)])
    b = np.array([comb(truths, i) * f ** i * (1 - f) ** (truths - i) for i in range(min(truths, kappa - 1) + 1)])
    return np.convolve(a, b)[:kappa] if kappa > 0 else np.zeros(0)


def make_model(T: int, q: float, f: float, kappa: int, replaceable: bool):
    @lru_cache(maxsize=None)
    def step(age: int, lies: int, a: int) -> tuple[float, float]:
        """(expected damage now, P(still served after this interaction | served now))."""
        w = d_dist(lies, age - lies, q, f, kappa)
        alive_now = w.sum()
        if alive_now <= 0:
            return 0.0, 0.0
        b = np.zeros(kappa)
        b[:len(w)] = w / alive_now
        pf = q if a else f
        p_block = b[kappa - 1] * pf
        return ((1 - q) if a else 0.0), 1.0 - p_block

    @lru_cache(maxsize=None)
    def V(t: int, age: int, lies: int) -> tuple[float, int]:
        if t == T:
            return 0.0, 0
        best = (-1.0, 0)
        for a in (0, 1):
            gain, p_alive = step(age, lies, a)
            val = gain + p_alive * V(t + 1, age + 1, lies + a)[0]
            if replaceable:
                val += (1 - p_alive) * V(t + 1, 0, 0)[0]
            if val > best[0] + 1e-12:
                best = (val, a)
        return best

    return V, step


def evaluate(T, q, f, kappa, replaceable, rule) -> float:
    """Exact value of a fixed rule lie(age, lies) by forward propagation."""
    V_dist = {(0, 0): 1.0}
    _, step = make_model(T, q, f, kappa, replaceable)
    total = 0.0
    for _ in range(T):
        new: dict = {}
        for (age, lies), p in V_dist.items():
            a = rule(age, lies)
            gain, p_alive = step(age, lies, a)
            total += p * gain
            new[(age + 1, lies + a)] = new.get((age + 1, lies + a), 0.0) + p * p_alive
            if replaceable and p_alive < 1:
                new[(0, 0)] = new.get((0, 0), 0.0) + p * (1 - p_alive)
        V_dist = new
    return total


def monte_carlo(T, q, f, kappa, replaceable, V, n=20000, seed=0) -> float:
    rng = np.random.default_rng(seed)
    tot = 0
    for _ in range(n):
        age = lies = d = 0
        for t in range(T):
            a = V(t, age, lies)[1]
            flag = rng.random() < (q if a else f)
            if a and not flag:
                tot += 1
            age, lies, d = age + 1, lies + a, d + int(flag)
            if d >= kappa:
                if not replaceable:
                    break
                age = lies = d = 0
    return tot / n


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, traces = [], []
    T, h = 40, 0.9
    for replaceable in (True, False):
        for u in (0.1, 0.3, 0.6, 1.0):
            for kappa in (1, 3, 5):
                q, f = u * h, 0.0
                V, _ = make_model(T, q, f, kappa, replaceable)
                v = V(0, 0, 0)[0]
                always = evaluate(T, q, f, kappa, replaceable, lambda age, l: 1)
                budget = evaluate(T, q, f, kappa, replaceable, lambda age, l, k=kappa: int(l < k - 1))
                # optimal actions along the path on which the first identity is never blocked
                lies, first = 0, None
                for t in range(T):
                    a = V(t, t, lies)[1]
                    traces.append({"replaceable": replaceable, "u": u, "kappa": kappa, "t": t, "lie": a})
                    if a and first is None:
                        first = t
                    lies += a
                rows.append({"replaceable": replaceable, "u": u, "h": h, "kappa": kappa, "T": T,
                             "optimal": v, "always_lie": always, "budget_kappa_minus_1": budget,
                             "optimal_minus_always": v - always, "first_lie": first, "lies_first_identity": lies,
                             "monte_carlo": monte_carlo(T, q, f, kappa, replaceable, V, n=4000)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dp_best_response.csv", index=False)
    pd.DataFrame(traces).to_csv(OUT / "dp_policies.csv", index=False)
    print(df.round(3).to_string())


if __name__ == "__main__":
    main()
