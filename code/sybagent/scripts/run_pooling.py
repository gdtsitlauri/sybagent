"""Study 20b: evidence pooling against the budget attacker (and the others), and the RL attacker against it.

Committee sharing (two-sided verdicts) with and without pooling of single disagreements, for committee capacities 5
and 10 per round; attackers lie 30%, always lie, budget (kappa - 1 lies per victim); false reports on. The worst
case over attackers is what a defender can count on.

    python scripts/run_pooling.py --out results/20_shared_defense/pooling --workers 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_shared as rs  # noqa: E402
from sybagent.config import from_dict  # noqa: E402
from sybagent.simulation import Simulation  # noqa: E402

SYBILS = [25, 50, 100]
POOL = [False, True]
CAPACITY = [5, 10]
SEEDS = [0, 1, 2]
EVIDENCE = "transcripts"           # members re-check the reporters' signed transcripts (pooling_probes/: probes)


def run_one(job):
    keys, cfg = job
    return {**keys, "seed": cfg["seed"], **rs.summarize(Simulation(from_dict(cfg)).run())}


def rl_job(job):
    b, pool, out = job
    params = {"share": "committee", "pool_evidence": pool, "evidence_mode": EVIDENCE}
    best = None
    for g in rs.GAMMAS:
        q, hist = "", []
        for ep in range(rs.EPISODES):
            adv = {"policy": "rl", "policy_train": True, "policy_q": q, "policy_gamma": g}
            sim = Simulation(from_dict(rs.config(b, 1000 + ep, adv, params, 1)))
            df = sim.run()
            q = str(Path(out) / "rl" / f"B{b}_pool{int(pool)}_g{g}.json")
            sim.strategy.save(q)
            hist.append(df["wrong_used_rate"].mean())
        score = sum(hist[rs.EPISODES // 2:]) / (rs.EPISODES - rs.EPISODES // 2)
        if best is None or score > best[0]:
            best = (score, g, q, sim.strategy.greedy_policy())
    _, g, q, pol = best
    (Path(out) / "rl" / f"B{b}_pool{int(pool)}_policy.json").write_text(json.dumps({"gamma": g, "policy": pol}))
    return [{"n_sybils": b, "pool_evidence": pool, "committee_capacity": 5, "attacker": "rl", "rl_gamma": g,
             "seed": s, **rs.summarize(Simulation(from_dict(rs.config(b, s, {"policy": "rl", "policy_q": q},
                                                                       params, 1))).run())} for s in SEEDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "rl").mkdir(parents=True, exist_ok=True)
    jobs = []
    for b, pool, cap, att, s in itertools.product(SYBILS, POOL, CAPACITY, rs.ATTACKERS, SEEDS):
        keys = {"n_sybils": b, "pool_evidence": pool, "committee_capacity": cap, "attacker": att}
        params = {"share": "committee", "pool_evidence": pool, "committee_capacity": cap, "evidence_mode": EVIDENCE}
        jobs.append((keys, rs.config(b, s, rs.ATTACKERS[att], params, 1)))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rl_futs = [ex.submit(rl_job, (b, pool, str(out))) for b in (25, 50) for pool in POOL]
        rows = list(ex.map(run_one, jobs, chunksize=2))
        for f in rl_futs:
            rows += f.result()
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
