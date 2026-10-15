"""Study 12: multi-criteria analysis of the defense's parameters (Pareto front).

Objectives, all minimized, as means over seeds of the last 20 rounds (cost over the whole run):
  damage    wrong answers used per answer          (tail_wrong_used_rate)
  cost      extra anchor queries per task          (final_audit_calls / tasks)
  exposure  Sybil share of the targets' peers      (tail_target_sybil_share)
  guarantee honest agents without an honest anchor (tail_honest_no_honest_anchor_frac), the quantity of Theorem 1
Reported per threat level B: the non-dominated configurations, the knee (closest to the ideal point after
normalization), the hypervolume of the front, and an epsilon-constraint table (least damage for a cost budget).

    python analysis/pareto.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
S12 = ROOT / "results" / "12_pareto"
OBJ = ["damage", "cost", "exposure", "no_honest_anchor"]
KEYS = ["defense.params.n_anchors", "defense.params.audit_prob", "defense.params.audit_rule",
        "defense.params.audit_anchors"]
BUDGETS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def load() -> pd.DataFrame:
    s = pd.read_csv(S12 / "summary.csv")
    tasks = 200 * 100
    s["damage"] = s["tail_wrong_used_rate"]
    s["cost"] = s["final_audit_calls"] / tasks
    s["exposure"] = s["tail_target_sybil_share"]
    s["quality"] = s["tail_mean_quality"]
    s["no_honest_anchor"] = s["tail_honest_no_honest_anchor_frac"]
    g = s.groupby(["adversary.n_sybils"] + KEYS)[OBJ + ["quality"]]
    m = g.mean().reset_index()
    se = g.sem().reset_index()
    for c in OBJ:
        m[c + "_se"] = se[c]
    return m


def non_dominated(x: np.ndarray) -> np.ndarray:
    keep = np.ones(len(x), dtype=bool)
    for i in range(len(x)):
        if not keep[i]:
            continue
        dom = np.all(x <= x[i], axis=1) & np.any(x < x[i], axis=1)
        if dom.any():
            keep[i] = False
    return keep


def hypervolume(front: np.ndarray, ref: np.ndarray, n: int = 400_000, seed: int = 0) -> float:
    """Monte Carlo hypervolume dominated by `front` inside the box [ideal, ref]."""
    rng = np.random.default_rng(seed)
    lo = front.min(axis=0)
    pts = lo + rng.random((n, front.shape[1])) * (ref - lo)
    dominated = np.zeros(n, dtype=bool)
    for p in front:
        dominated |= np.all(pts >= p, axis=1)
    return float(dominated.mean() * np.prod(ref - lo))


def main() -> None:
    m = load()
    fronts, knees, eps, hv = [], [], [], []
    for b, g in m.groupby("adversary.n_sybils"):
        x = g[OBJ].to_numpy()
        nd = non_dominated(x)
        f = g[nd].copy()
        lo, hi = x.min(0), x.max(0)
        z = (f[OBJ].to_numpy() - lo) / np.where(hi > lo, hi - lo, 1)
        f["dist_to_ideal"] = np.linalg.norm(z, axis=1)
        fronts.append(f.sort_values("cost"))
        knees.append(f.loc[f["dist_to_ideal"].idxmin()].copy())
        # knee among the configurations that keep Theorem 1 (anchors not audited by anchors)
        safe = f[~f["defense.params.audit_anchors"].astype(bool)]
        k = safe.loc[safe["dist_to_ideal"].idxmin()].copy()
        k["subset"] = "keeps_theorem1"
        knees[-1]["subset"] = "all"
        knees.append(k)
        ref = hi * 1.05 + 1e-9
        hv.append({"n_sybils": b, "configs": len(g), "front_size": int(nd.sum()),
                   "hypervolume": hypervolume(f[OBJ].to_numpy(), ref), "box_volume": float(np.prod(ref - lo))})
        for c in BUDGETS:
            ok = g[g["cost"] <= c + 1e-12]
            if len(ok):
                r = ok.loc[ok["damage"].idxmin()]
                eps.append({"n_sybils": b, "cost_budget": c, **{k: r[k] for k in KEYS},
                            "damage": r["damage"], "cost": r["cost"], "exposure": r["exposure"]})
    pd.concat(fronts).to_csv(S12 / "pareto_front.csv", index=False)
    pd.DataFrame(knees).to_csv(S12 / "knee.csv", index=False)
    pd.DataFrame(eps).to_csv(S12 / "epsilon_constraint.csv", index=False)
    pd.DataFrame(hv).to_csv(S12 / "hypervolume.csv", index=False)
    m.to_csv(S12 / "config_means.csv", index=False)
    print(pd.concat(fronts).round(3).to_string(), pd.DataFrame(knees).round(3).to_string(),
          pd.DataFrame(eps).round(3).to_string(), pd.DataFrame(hv).round(4).to_string(), sep="\n\n")


if __name__ == "__main__":
    main()
