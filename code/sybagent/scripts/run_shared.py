"""Study 20: shared defenses against Sybils that lie, report falsely and vote strategically.

Three parts, all written to results/20_shared_defense/:
  main/       B x sharing mode x attacker x false reports (3 seeds)
  threshold/  committee quorum x committee size x anchors x B, always-lying Sybils with false reports (3 seeds):
              checks the safety/liveness threshold of the committee theorem
  rl/         Q-learning attacker (gamma chosen by training damage, as in study 11) vs local and committee sharing

    python scripts/run_shared.py --out results/20_shared_defense --workers 6
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

# committee members judge signed transcripts (final design); results/20_shared_defense/probes/ used probes
EVIDENCE = "transcripts"

BASE = {
    "rounds": 100,
    "network": {"n_honest": 200},
    "adversary": {"n_targets": 5, "attractiveness": 0.5, "service_quality": 0.85, "adaptive_cards": True,
                  "coordinated_lies": True, "answer_wrong_prob": 1.0},
    "selection": {"kind": "reputation"},
    "defense": {"kind": "shared", "params": {"n_anchors": 3, "audit_prob": 0.3, "audit_flag": 3,
                                             "audit_rule": "unanimous", "evidence_mode": EVIDENCE}},
}
SYBILS = [25, 50, 100, 200]
MODES = ["local", "naive", "threshold", "alerts", "committee", "alerts+committee"]
ATTACKERS = {"lie1.0": {"policy": "fixed", "answer_wrong_prob": 1.0},
             "lie0.3": {"policy": "fixed", "answer_wrong_prob": 0.3},
             "budget": {"policy": "budget"}}
FALSE_REPORTS = [0, 1]
SEEDS = [0, 1, 2]
QUORUMS = [0.5, 2 / 3, 0.8, 0.9]
COMMITTEES = [15, 45]
ANCHORS = [3, 6]
GAMMAS, EPISODES = [0.0, 0.5, 0.9], 8


def config(b: int, seed: int, adv: dict, params: dict, false_reports: int) -> dict:
    c = copy.deepcopy(BASE)
    c["seed"] = seed
    c["adversary"]["n_sybils"] = b
    c["adversary"]["false_reports"] = false_reports
    c["adversary"].update(adv)
    c["defense"]["params"].update(params)
    return c


def summarize(df: pd.DataFrame) -> dict:
    tail = df.tail(20).mean(numeric_only=True)
    last = df.iloc[-1]
    out = {"wrong_used": tail["wrong_used_rate"], "wrong_used_all": df["wrong_used_rate"].mean(),
           "target_sybil_share": tail["target_sybil_share"],
           "no_honest_anchor": tail["honest_no_honest_anchor_frac"],
           "queries_per_task": last["audit_calls"] / (200 * len(df))}
    for k in ["global_blocks_sybil", "global_blocks_honest", "reports_honest", "reports_false", "reports_muted",
              "honest_reporters_muted", "committees"]:
        out[k] = last.get(f"shared_{k}", float("nan"))
    if "strategy_lie_frac" in df:
        out["lie_frac"] = last["strategy_lie_frac"]
    return out


def run_one(job: tuple) -> dict:
    part, keys, cfg = job
    df = Simulation(from_dict(cfg)).run()
    return {"part": part, **keys, "seed": cfg["seed"], **summarize(df)}


def rl_job(job: tuple) -> list[dict]:
    b, mode, fr, out = job
    best = None
    for g in GAMMAS:
        q, hist = "", []
        for ep in range(EPISODES):
            adv = {"policy": "rl", "policy_train": True, "policy_q": q, "policy_gamma": g}
            sim = Simulation(from_dict(config(b, 1000 + ep, adv, {"share": mode}, fr)))
            df = sim.run()
            q = str(Path(out) / "rl" / f"B{b}_{mode}_fr{fr}_g{g}.json")
            sim.strategy.save(q)
            hist.append(df["wrong_used_rate"].mean())
        score = sum(hist[EPISODES // 2:]) / (EPISODES - EPISODES // 2)
        if best is None or score > best[0]:
            best = (score, g, q, sim.strategy.greedy_policy())
    _, g, q, pol = best
    (Path(out) / "rl" / f"B{b}_{mode}_fr{fr}_policy.json").write_text(json.dumps({"gamma": g, "policy": pol}))
    rows = []
    for seed in SEEDS:
        df = Simulation(from_dict(config(b, seed, {"policy": "rl", "policy_q": q}, {"share": mode}, fr))).run()
        rows.append({"part": "rl", "n_sybils": b, "share": mode, "attacker": "rl", "false_reports": fr,
                     "rl_gamma": g, "seed": seed, **summarize(df)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "rl").mkdir(parents=True, exist_ok=True)
    jobs = []
    for b, mode, att, fr, s in itertools.product(SYBILS, MODES, ATTACKERS, FALSE_REPORTS, SEEDS):
        keys = {"n_sybils": b, "share": mode, "attacker": att, "false_reports": fr}
        jobs.append(("main", keys, config(b, s, ATTACKERS[att], {"share": mode}, fr)))
    for b, qu, c, a, s in itertools.product([50, 100, 200], QUORUMS, COMMITTEES, ANCHORS, SEEDS):
        keys = {"n_sybils": b, "quorum": round(qu, 3), "committee_size": c, "n_anchors": a}
        jobs.append(("threshold", keys, config(b, s, ATTACKERS["lie1.0"], {"share": "committee", "quorum": qu,
                                                                           "committee_size": c, "n_anchors": a}, 1)))
    rl_jobs = [(b, mode, 1, str(out)) for b in (50, 200) for mode in ("local", "committee")]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rl_futs = [ex.submit(rl_job, j) for j in rl_jobs]
        rows = list(ex.map(run_one, jobs, chunksize=2))
        for f in rl_futs:
            rows += f.result()
    df = pd.DataFrame(rows)
    for part, g in df.groupby("part"):
        g.dropna(axis=1, how="all").to_csv(out / f"{part}.csv", index=False)
    (out / "config.json").write_text(json.dumps({"base": BASE, "sybils": SYBILS, "modes": MODES,
                                                 "attackers": ATTACKERS, "false_reports": FALSE_REPORTS,
                                                 "seeds": SEEDS, "quorums": QUORUMS, "committees": COMMITTEES,
                                                 "anchors": ANCHORS, "gammas": GAMMAS, "episodes": EPISODES},
                                                indent=2))
    print(f"{len(df)} evaluation runs -> {out}")


if __name__ == "__main__":
    main()
