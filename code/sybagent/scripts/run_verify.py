"""Study 22: verifiable answers. Proof-carrying evidence and verified reputation, beyond the committee threshold.

Each served answer is verified with certainty with probability v (a proof, a test, a recomputation), at a cost.
Modes: correct (verified answers are corrected only), ban (a statistically implausible count of verified errors in
the last 20 verified answers blocks the agent network-wide, no vote), ban+rep (also: selection favours agents with a
good verified record). Attackers: always lie, lie 30%, budget, sleeper (truthful until round 50); an RL attacker
against ban+rep. No committees and no sharing otherwise, to isolate the effect.

    python scripts/run_verify.py --out results/22_verifiable --workers 6
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

SYBILS = [50, 200, 400]
VERIFY = [0.0, 0.02, 0.05, 0.1, 0.2]
MODES = {"correct": {"proof_ban": False, "verified_rep": False},
         "ban": {"proof_ban": True, "verified_rep": False},
         "ban+rep": {"proof_ban": True, "verified_rep": True}}
ATTACKERS = {**rs.ATTACKERS, "sleeper": {"policy": "sleeper", "sleeper_round": 50}}
SEEDS = [0, 1, 2]


def params(v: float, mode: str) -> dict:
    return {"share": "local", "verify_prob": v, **MODES[mode]}


def summarize(df: pd.DataFrame) -> dict:
    out = rs.summarize(df)
    last = df.iloc[-1]
    out["verifications_per_task"] = last["verify_calls"] / (200 * len(df))
    out["proof_bans"] = last["shared_proof_bans"]
    return out


def run_one(job):
    keys, cfg = job
    return {**keys, "seed": cfg["seed"], **summarize(Simulation(from_dict(cfg)).run())}


def rl_job(job):
    b, v, mode, out = job
    best = None
    for g in rs.GAMMAS:
        q, hist = "", []
        for ep in range(rs.EPISODES):
            adv = {"policy": "rl", "policy_train": True, "policy_q": q, "policy_gamma": g}
            sim = Simulation(from_dict(rs.config(b, 1000 + ep, adv, params(v, mode), 0)))
            df = sim.run()
            q = str(Path(out) / "rl" / f"B{b}_v{v}_{mode}_g{g}.json")
            sim.strategy.save(q)
            hist.append(df["wrong_used_rate"].mean())
        score = sum(hist[rs.EPISODES // 2:]) / (rs.EPISODES - rs.EPISODES // 2)
        if best is None or score > best[0]:
            best = (score, g, q, sim.strategy.greedy_policy())
    _, g, q, pol = best
    (Path(out) / "rl" / f"B{b}_v{v}_{mode}_policy.json").write_text(json.dumps({"gamma": g, "policy": pol}))
    return [{"n_sybils": b, "verify_prob": v, "mode": mode, "attacker": "rl", "rl_gamma": g, "seed": s,
             **summarize(Simulation(from_dict(rs.config(b, s, {"policy": "rl", "policy_q": q}, params(v, mode),
                                                        0))).run())} for s in SEEDS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "rl").mkdir(parents=True, exist_ok=True)
    jobs = []
    for b, v, mode, att, s in itertools.product(SYBILS, VERIFY, MODES, ATTACKERS, SEEDS):
        if v == 0.0 and mode != "correct":
            continue
        keys = {"n_sybils": b, "verify_prob": v, "mode": mode, "attacker": att}
        jobs.append((keys, rs.config(b, s, ATTACKERS[att], params(v, mode), 0)))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rl_futs = [ex.submit(rl_job, (b, 0.05, "ban+rep", str(out))) for b in (200, 400)]
        rows = list(ex.map(run_one, jobs, chunksize=2))
        for f in rl_futs:
            rows += f.result()
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"sybils": SYBILS, "verify": VERIFY, "modes": MODES,
                                                 "attackers": ATTACKERS, "seeds": SEEDS}, indent=2))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
