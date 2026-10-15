"""Empirical checks of Proposition 2 (O(m) Sybils eclipse semantic top-k, independent of N) and Theorem 1."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "sybagent" / "src"))

from sybagent.config import SimConfig  # noqa: E402
from sybagent.defenses.proposed import anchor_eclipse_bound  # noqa: E402
from sybagent.metrics import time_to_eclipse  # noqa: E402
from sybagent.simulation import Simulation  # noqa: E402


def prop2(n_honest: int, defense: str, seed: int, rounds: int = 60) -> dict:
    cfg = SimConfig(seed=seed, rounds=rounds)
    cfg.network.n_honest = n_honest
    cfg.network.max_neighbors = 8
    cfg.adversary.n_sybils = 8                # B = m
    cfg.adversary.n_targets = 1
    cfg.adversary.attractiveness = 1.0
    cfg.adversary.card_jitter = 0.0
    cfg.adversary.adaptive_cards = True
    cfg.adversary.service_quality = 0.85
    cfg.selection.kind = "similarity"
    cfg.selection.temperature = 0.0           # deterministic top-k
    cfg.selection.experience_weight = 0.0
    cfg.defense.kind = defense
    if defense == "proposed":
        cfg.defense.params = {"n_anchors": 3}
    df = Simulation(cfg).run()
    return {
        "n_honest": n_honest, "defense": defense, "seed": seed,
        "time_to_eclipse_all_targets": time_to_eclipse(df.to_dict("records")),
        "tail_targets_eclipsed_frac": df.tail(20)["targets_eclipsed_frac"].mean(),
        "tail_target_sybil_share": df.tail(20)["target_sybil_share"].mean(),
        "tail_target_min_honest_neighbors": df.tail(20)["target_min_honest_neighbors"].mean(),
    }


def main() -> None:
    rows = [prop2(n, d, s) for n in [100, 200, 400, 800] for d in ["none", "proposed"] for s in range(5)]
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "7_theory_check"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "prop2.csv", index=False)
    agg = df.groupby(["defense", "n_honest"]).mean(numeric_only=True).drop(columns="seed").round(3)
    agg["theorem1_bound_a3"] = [anchor_eclipse_bound(n, 8, 3) if d == "proposed" else float("nan")
                                for d, n in agg.index]
    print(agg.to_string())
    (ROOT / "tables").mkdir(exist_ok=True)
    (ROOT / "tables" / "prop2.md").write_text(agg.to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    main()
