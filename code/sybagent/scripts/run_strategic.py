"""Study 11: strategic (learning) attacker against the defender's audit commitments.

For every defender commitment (audit rate u, audit anchors too, decision rule) and Sybil count B:
  * fixed attackers: lie with prob. 0.3 or always (coordinated lies);
  * budget attacker: knows the defense, lies to each victim at most audit_flag - 1 times, always lies as anchor;
  * RL attacker: tabular Q-learning, trained in the simulator for EPISODES episodes per discount factor gamma in
    GAMMAS; the attacker keeps the gamma with the highest training damage (last half of training), then the frozen
    greedy policy is evaluated on the evaluation seeds.

    python scripts/run_strategic.py --out results/11_strategic_attacker --workers 6
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sybagent.config import from_dict  # noqa: E402
from sybagent.simulation import Simulation  # noqa: E402

BASE = {
    "rounds": 100,
    "network": {"n_honest": 200},
    "adversary": {"n_targets": 5, "attractiveness": 0.5, "service_quality": 0.85, "adaptive_cards": True,
                  "coordinated_lies": True, "answer_wrong_prob": 1.0},
    "selection": {"kind": "reputation"},
    "defense": {"kind": "proposed", "params": {"n_anchors": 3, "audit_flag": 3}},
}
SYBILS = [50, 200]
AUDIT = [0.0, 0.1, 0.3, 0.6, 1.0]
AUDIT_ANCHORS = [False, True]
RULES = ["majority", "unanimous"]
FIXED = {"lie0.3": {"policy": "fixed", "answer_wrong_prob": 0.3},
         "lie1.0": {"policy": "fixed", "answer_wrong_prob": 1.0},
         "budget": {"policy": "budget"}}
GAMMAS = [0.0, 0.5, 0.9]
EPISODES = 8
TRAIN_SEED0 = 1000
EVAL_SEEDS = [0, 1, 2]


def config(b: int, u: float, aa: bool, rule: str, seed: int, adv: dict) -> dict:
    c = copy.deepcopy(BASE)
    c["seed"] = seed
    c["adversary"]["n_sybils"] = b
    c["adversary"].update(adv)
    c["defense"]["params"].update(audit_prob=u, audit_anchors=aa, audit_rule=rule)
    return c


def summarize(df: pd.DataFrame) -> dict:
    m = df.mean(numeric_only=True)
    tasks = 200 * len(df)
    return {"wrong_used": m["wrong_used_rate"], "wrong_served": m["wrong_answer_rate"],
            "audit_queries_per_task": df["audit_calls"].iloc[-1] / tasks,
            "target_sybil_share": m["target_sybil_share"], "mean_quality": m["mean_quality"],
            "blocked_honest_pairs": df["blocked_honest_pairs"].iloc[-1],
            "blocked_sybil_pairs": df["blocked_sybil_pairs"].iloc[-1],
            "bad_corrections": m["bad_corrections"], "good_corrections": m["good_corrections"],
            "lie_frac": m.get("strategy_lie_frac", float("nan"))}


def run_fixed(job: tuple) -> dict:
    b, u, aa, rule, name, seed, out = job
    cfg = config(b, u, aa, rule, seed, FIXED[name])
    df = Simulation(from_dict(cfg)).run()
    tag = f"B{b}_u{u}_aa{int(aa)}_{rule}_{name}_s{seed}"
    df.to_csv(Path(out) / "runs" / f"{tag}.csv", index=False)
    return {"n_sybils": b, "audit_prob": u, "audit_anchors": aa, "audit_rule": rule, "attacker": name,
            "seed": seed, **summarize(df)}


def run_rl(job: tuple) -> list[dict]:
    b, u, aa, rule, out = job
    out = Path(out)
    best = None
    curves = []
    for g in GAMMAS:
        q = ""
        hist = []
        for ep in range(EPISODES):
            adv = {"policy": "rl", "policy_train": True, "policy_q": q, "policy_gamma": g}
            sim = Simulation(from_dict(config(b, u, aa, rule, TRAIN_SEED0 + ep, adv)))
            df = sim.run()
            q = str(out / "q" / f"B{b}_u{u}_aa{int(aa)}_{rule}_g{g}.json")
            sim.strategy.save(q)
            hist.append(float(df["wrong_used_rate"].mean()))
            curves.append({"n_sybils": b, "audit_prob": u, "audit_anchors": aa, "audit_rule": rule, "gamma": g,
                           "episode": ep, "train_wrong_used": hist[-1]})
        score = sum(hist[EPISODES // 2:]) / (EPISODES - EPISODES // 2)
        if best is None or score > best[0]:
            best = (score, g, q, sim.strategy.greedy_policy())
    _, g, q, pol = best
    (out / "q" / f"B{b}_u{u}_aa{int(aa)}_{rule}_policy.json").write_text(json.dumps({"gamma": g, "policy": pol}))
    rows = []
    for seed in EVAL_SEEDS:
        df = Simulation(from_dict(config(b, u, aa, rule, seed, {"policy": "rl", "policy_q": q}))).run()
        df.to_csv(out / "runs" / f"B{b}_u{u}_aa{int(aa)}_{rule}_rl_s{seed}.csv", index=False)
        rows.append({"n_sybils": b, "audit_prob": u, "audit_anchors": aa, "audit_rule": rule, "attacker": "rl",
                     "seed": seed, "rl_gamma": g, **summarize(df)})
    return rows + [{"_curve": c} for c in curves]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "q").mkdir(parents=True, exist_ok=True)
    defenders = list(itertools.product(SYBILS, AUDIT, AUDIT_ANCHORS, RULES))
    fixed_jobs = [(b, u, aa, r, n, s, str(out)) for (b, u, aa, r) in defenders for n in FIXED for s in EVAL_SEEDS]
    rl_jobs = [(b, u, aa, r, str(out)) for (b, u, aa, r) in defenders]
    rows, curves = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rl_futs = [ex.submit(run_rl, j) for j in rl_jobs]       # long jobs first
        rows += list(ex.map(run_fixed, fixed_jobs, chunksize=1))
        for f in rl_futs:
            for r in f.result():
                (curves if "_curve" in r else rows).append(r.get("_curve", r))
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "rl_training.csv", index=False)
    (out / "config.json").write_text(json.dumps({"base": BASE, "sybils": SYBILS, "audit": AUDIT,
                                                 "audit_anchors": AUDIT_ANCHORS, "rules": RULES, "fixed": FIXED,
                                                 "gammas": GAMMAS, "episodes": EPISODES,
                                                 "eval_seeds": EVAL_SEEDS}, indent=2))
    print(f"{len(rows)} evaluation runs -> {out}")


if __name__ == "__main__":
    main()
