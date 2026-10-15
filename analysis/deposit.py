"""Deposit calculator: the identity cost that makes an attack unprofitable on the real ERC-8004 registries.

An attacker gains V from an attack and pays a deposit d per identity it registers (forfeited or locked). The attack
does not pay when d > V / B_needed, where B_needed is the number of identities the attack requires:
  * reputation discovery, no anchors: the median number of identities to take all top-5 places of a category
    (digital twin, study 14);
  * with a anchors: the Sybil agents for an eclipse probability eps = 0.5 (Theorem 1, inverted), for a = 1..6;
  * collective defense: the Sybil count at which committees stop being possible (Corollary 2, twin).
The output is the deposit per unit of attack value (d / V) for each case; the ratio between cases does not depend on
V, which is unknown.

    python analysis/deposit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TWIN = ROOT / "results" / "14_digital_twin"


def anchors_needed(n: int, a: int, eps: float = 0.5) -> float:
    r = eps ** (1.0 / a)
    return r * (n - a) / (1 - r)


def main() -> None:
    lim = pd.read_csv(TWIN / "committee_limits.csv")
    rows = []
    for chain in ("ethereum", "base"):
        n = int(json.loads((TWIN / chain / "summary.json").read_text())["registrations"])
        rk = pd.read_csv(TWIN / chain / "risk_over_time.csv")
        rep = float(rk[rk["t"] == rk["t"].max()]["reputation_identities"].median())
        rows.append({"chain": chain, "registered": n, "case": "reputation discovery, no anchors",
                     "identities_needed": rep, "deposit_per_value": 1 / rep})
        for a in range(1, 7):
            b = anchors_needed(n, a)
            rows.append({"chain": chain, "registered": n, "case": f"{a} anchors, eclipse probability 0.5",
                         "identities_needed": b, "deposit_per_value": 1 / b})
        c = lim[(lim["chain"] == chain) & (lim["anchors"] == 3)]["sybils_at_rho_half"].iloc[0]
        rows.append({"chain": chain, "registered": n, "case": "break committees (3 anchors)",
                     "identities_needed": float(c), "deposit_per_value": 1 / c})
    df = pd.DataFrame(rows)
    base = df[df["case"] == "reputation discovery, no anchors"].set_index("chain")["identities_needed"]
    df["identities_vs_no_anchors"] = [r.identities_needed / base[r.chain] for r in df.itertuples()]
    df.to_csv(TWIN / "deposit.csv", index=False)
    print(df.round(6).to_string())


if __name__ == "__main__":
    main()
