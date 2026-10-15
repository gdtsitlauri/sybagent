"""Committee planning on the real ERC-8004 registries (digital twin + Theorem 2).

For the registry sizes mirrored by the twin (study 14) and a range of Sybil identity counts B:
  * effective corruption rho = p_s + (1 - p_s) * pi, with pi = P(Bin(a, p) > a/2);
  * the smallest committee size c (quorum chosen in [0.5, 0.95]) whose exact binomial error probabilities meet a target
    (framing <= 1e-6, missing a lying Sybil <= 1e-3), or "impossible";
  * the Sybil count at which committees stop being possible (rho = 1/2), in identities;
  * the on-chain gas of one judgement at that committee size, from study 21 (linear fit in c).
Model parameters of an honest member's verdict (r = 3 probes, block if >= 2 disagree, honest answer accuracy
0.95): eps = P(>= 2 of 3 probes wrongly disagree with an honest agent), delta = P(>= 2 of 3 probes expose a
lying Sybil) when the member's anchor majority is honest.

    python analysis/twin_committee.py
"""

from __future__ import annotations

import json
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "14_digital_twin"
Q = 2 / 3
FRAME_TARGET, MISS_TARGET = 1e-6, 1e-3
ACC = 0.95


def member_rates(a: int, p: float) -> tuple[float, float, float]:
    pi = float(binom.sf(a // 2, a, p))                   # P(Sybil majority among a anchors), a odd
    if a % 2 == 0:
        pi = float(binom.sf(a // 2 - 1, a, p))           # ties count as a Sybil-controlled verdict (conservative)
    per_probe_false = (1 - ACC) * ACC                    # honest x wrong, honest majority right
    eps = float(binom.sf(1, 3, per_probe_false))
    per_probe_detect = ACC                               # honest majority gives the right answer
    delta = float(binom.sf(1, 3, per_probe_detect))
    return pi, eps, delta


def errors(c: int, rho_s: float, rho_l: float, q: float = Q) -> tuple[float, float]:
    need = ceil(q * c)
    return float(binom.sf(need - 1, c, rho_s)), float(binom.cdf(need - 1, c, rho_l))


def main() -> None:
    gas = pd.read_csv(ROOT / "results" / "21_onchain_blocklist" / "gas.csv")
    g = gas.groupby("committee_size")["total_gas"].mean()
    slope = (g.iloc[-1] - g.iloc[0]) / (g.index[-1] - g.index[0])
    intercept = g.iloc[0] - slope * g.index[0]
    sizes = {}
    for chain in ("ethereum", "base"):
        s = json.loads((OUT / chain / "summary.json").read_text())
        sizes[chain] = int(s["registrations"])
    rows, limits = [], []
    for chain, n in sizes.items():
        for a in (3, 5, 7):
            # Sybil count at which rho reaches 1/2 (bisection on B)
            lo, hi = 0, 20 * n
            for _ in range(60):
                mid = (lo + hi) / 2
                p_s = mid / (n + mid)
                pi, _, _ = member_rates(a, p_s)
                if p_s + (1 - p_s) * pi < 0.5:
                    lo = mid
                else:
                    hi = mid
            limits.append({"chain": chain, "registered": n, "anchors": a, "sybils_at_rho_half": int(lo),
                           "sybil_share_at_rho_half": lo / (n + lo)})
            for frac in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4):
                b = int(frac * n / (1 - frac))
                p_s = b / (n + b)
                pi, eps, delta = member_rates(a, p_s)
                rho = p_s + (1 - p_s) * pi
                rho_s, rho_l = rho + (1 - rho) * eps, (1 - rho) * delta
                c_ok, q_ok = None, None
                for q in sorted({Q, *np.round(np.arange(0.5, 0.951, 0.025), 3)}):
                    if not rho_s < q < rho_l:
                        continue
                    for c in range(3, 2001):
                        fr, ms = errors(c, rho_s, rho_l, q)
                        if fr <= FRAME_TARGET and ms <= MISS_TARGET:
                            if c_ok is None or c < c_ok:
                                c_ok, q_ok = c, q
                            break
                rows.append({"chain": chain, "registered": n, "anchors": a, "sybil_share": p_s, "sybils": b,
                             "pi": pi, "rho": rho, "rho_frame": rho_s, "rho_detect": rho_l,
                             "best_quorum": q_ok if c_ok else float("nan"),
                             "committee_size_needed": c_ok if c_ok else "impossible",
                             "gas_per_judgement": (intercept + slope * c_ok) if c_ok else float("nan")})
    df, lim = pd.DataFrame(rows), pd.DataFrame(limits)
    df.to_csv(OUT / "committee_planning.csv", index=False)
    lim.to_csv(OUT / "committee_limits.csv", index=False)
    print(df.round(4).to_string(), lim.to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
