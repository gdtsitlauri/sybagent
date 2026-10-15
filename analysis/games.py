"""Game-theoretic analysis (study 11): two Stackelberg games, the defender moves first.

Audit game. The defender commits to an audit rule (rate u, audit anchors or not, majority or unanimity); the attacker
best-responds with the most damaging of the strategies in study 11 (fixed lie rates, the knowledge-based budget
attacker, the RL attacker). The defender's loss is  damage + c * extra queries per task  for a price c per query.

Identity game. The attacker buys B identities at price rho / N each (rho = price of N identities in units of the value
of one eclipse) and gets P(no honest anchor) <= (B / (N + B - a))^a (Theorem 1). Its best response B*(a) maximizes
P - rho * B / N; the defender picks a. Evaluated for the simulator's N = 200 and for the real registry sizes.

    python analysis/games.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S11 = ROOT / "results" / "11_strategic_attacker"
OUT = ROOT / "results" / "11_strategic_attacker" / "games"
PRICES = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]


def audit_game() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(S11 / "summary.csv")
    keys = ["n_sybils", "audit_prob", "audit_anchors", "audit_rule"]
    mean = s.groupby(keys + ["attacker"])[["wrong_used", "audit_queries_per_task", "blocked_honest_pairs"]] \
        .mean().reset_index()
    # best response = attacker with the highest damage against each commitment
    br = mean.loc[mean.groupby(keys)["wrong_used"].idxmax()].rename(columns={"attacker": "best_response"})
    rl = mean[mean["attacker"] == "rl"].set_index(keys)["wrong_used"]
    fixed_best = mean[mean["attacker"] != "rl"].groupby(keys)["wrong_used"].max()
    br = br.set_index(keys)
    br["rl_minus_best_fixed"] = rl - fixed_best
    br = br.reset_index()
    eq, be = [], []
    # only commitments that keep Theorem 1 (anchors are never audited by anchors, see study 11)
    for b, g in br[~br["audit_anchors"].astype(bool)].groupby("n_sybils"):
        for rule, gr in g.groupby("audit_rule"):
            d0 = gr.loc[gr["audit_prob"] == 0.0, "wrong_used"].iloc[0]
            one = gr[gr["audit_prob"] == 1.0].iloc[0]
            # price per query at which u = 1 and u = 0 give the same loss (Proposition 7: the optimum is an end point)
            be.append({"n_sybils": b, "audit_rule": rule, "damage_u0": d0, "damage_u1": one["wrong_used"],
                       "queries_u1": one["audit_queries_per_task"],
                       "break_even_price": (d0 - one["wrong_used"]) / one["audit_queries_per_task"]})
        for c in PRICES:
            loss = g["wrong_used"] + c * g["audit_queries_per_task"]
            i = loss.idxmin()
            eq.append({"n_sybils": b, "price_per_query": c, **{k: g.loc[i, k] for k in keys[1:]},
                       "damage": g.loc[i, "wrong_used"], "queries_per_task": g.loc[i, "audit_queries_per_task"],
                       "loss": loss[i]})
    # linearity of damage in u under the best response (basis of the bang-bang result)
    lin = []
    for (b, aa, rule), g in br.groupby(["n_sybils", "audit_anchors", "audit_rule"]):
        x, y = g["audit_prob"].to_numpy(), g["wrong_used"].to_numpy()
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        lin.append({"n_sybils": b, "audit_anchors": aa, "audit_rule": rule, "slope": coef[0],
                    "intercept": coef[1], "r2": r2})
    pd.DataFrame(be).to_csv(OUT / "break_even.csv", index=False)
    return br, pd.DataFrame(eq), pd.DataFrame(lin)


def identity_game(n: int, rhos=(0.01, 0.1, 1.0), anchors=range(1, 9)) -> pd.DataFrame:
    rows = []
    grid = np.unique(np.concatenate([np.arange(0, 50 * n + 1, max(1, n // 200)), [0]]))
    for rho in rhos:
        for a in anchors:
            p = np.where(grid >= a, (grid / np.maximum(n + grid - a, 1)) ** a, 0.0)
            util = p - rho * grid / n
            i = int(np.argmax(util))
            rows.append({"N": n, "rho": rho, "anchors": a, "best_B": int(grid[i]), "best_B_over_N": grid[i] / n,
                         "p_eclipse": float(p[i]), "attacker_utility": float(util[i])})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    br, eq, lin = audit_game()
    br.to_csv(OUT / "best_responses.csv", index=False)
    eq.to_csv(OUT / "audit_equilibria.csv", index=False)
    lin.to_csv(OUT / "damage_linearity.csv", index=False)
    print(br.round(3).to_string(), eq.round(3).to_string(), lin.round(3).to_string(), sep="\n\n")
    sizes = {"simulator": 200}
    for chain in ("ethereum", "base"):
        f = ROOT / "results" / "14_digital_twin" / chain / "summary.json"
        if f.exists():
            sizes[chain] = int(json.loads(f.read_text())["registrations"])
    ig = pd.concat([identity_game(n).assign(network=k) for k, n in sizes.items()], ignore_index=True)
    ig.to_csv(OUT / "identity_game.csv", index=False)
    print(ig.round(4).to_string())


if __name__ == "__main__":
    main()
