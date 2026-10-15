"""Study 17: the defenses with real questions (MMLU) answered by a real LLM (analysis/build_real_answers.py).

Honest errors are the LLM's genuine errors (expert inside the agent's specialty, non-specialist outside it). The
statistical tests of committees and of the proof ban use the measured honest error rate (non-specialist error plus
a margin), because a 5% bound would ban honest agents for their genuine mistakes.

    python scripts/run_real_answers.py --out results/17_real_answers --workers 6
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_shared as rs  # noqa: E402
from sybagent.config import from_dict  # noqa: E402
from sybagent.simulation import Simulation  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = "data/real_answers/bundle.npz"
STATS = ROOT / "results" / "17_real_answers" / "bundle_stats.json"
SEEDS = [0, 1, 2]
ATTACKERS = {"lie1.0": {"policy": "fixed", "answer_wrong_prob": 1.0},
             "lie0.3": {"policy": "fixed", "answer_wrong_prob": 0.3}}


def defenses(honest_error: float) -> dict:
    stat = {"honest_error": honest_error}
    return {
        "none": {"kind": "none"},
        "anchors": {"kind": "shared", "params": {"share": "local", "audit_prob": 0.0}},
        "second_opinions_u0.3": {"kind": "shared", "params": {"share": "local", "audit_prob": 0.3}},
        "second_opinions_u1": {"kind": "shared", "params": {"share": "local", "audit_prob": 1.0}},
        "committees": {"kind": "shared", "params": {"share": "alerts+committee", "audit_prob": 0.3,
                                                    "evidence_mode": "transcripts", "pool_evidence": True,
                                                    "committee_capacity": 10, **stat}},
        "verify_0.05": {"kind": "shared", "params": {"share": "local", "audit_prob": 0.3, "verify_prob": 0.05,
                                                     "proof_ban": True, "verified_rep": True, **stat}},
        "verify_0.2": {"kind": "shared", "params": {"share": "local", "audit_prob": 0.3, "verify_prob": 0.2,
                                                    "proof_ban": True, "verified_rep": True, **stat}},
    }


def config(b: int, seed: int, att: str, defense: dict) -> dict:
    c = copy.deepcopy(rs.BASE)
    c["seed"] = seed
    c["network"].update(capability_dim=384, real_cards=BUNDLE)
    c["adversary"].update(n_sybils=b, adaptive_noise=0.02, **ATTACKERS[att])
    c["adversary"]["false_reports"] = 1 if defense.get("params", {}).get("share", "") != "local" else 0
    d = copy.deepcopy(defense)
    if d["kind"] == "shared":
        d["params"] = {**c["defense"]["params"], **d["params"]}
    c["defense"] = d
    return c


def run_one(job):
    keys, cfg = job
    df = Simulation(from_dict(cfg)).run()
    out = rs.summarize(df) if "shared_committees" in df else {
        "wrong_used": df.tail(20)["wrong_used_rate"].mean(), "wrong_used_all": df["wrong_used_rate"].mean(),
        "target_sybil_share": df.tail(20)["target_sybil_share"].mean(),
        "queries_per_task": df.iloc[-1]["audit_calls"] / (200 * len(df))}
    out["honest_wrong_served_baseline"] = df.tail(20)["wrong_answer_rate"].mean()
    out["verifications_per_task"] = df.iloc[-1].get("verify_calls", 0) / (200 * len(df))
    return {**keys, "seed": cfg["seed"], **out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    stats = json.loads(STATS.read_text())
    honest_error = min(0.9, (1 - stats["generic_accuracy"]) + 0.05)
    jobs = []
    for b, (name, d), att, s in itertools.product([0, 25, 50, 200], defenses(honest_error).items(), ATTACKERS,
                                                  SEEDS):
        if b == 0 and (name != "none" or att != "lie1.0"):
            continue
        cfg = config(b, s, att, d)
        if b == 0:
            cfg["adversary"]["enabled"] = False
        jobs.append(({"n_sybils": b, "defense": name, "attacker": att}, cfg))
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(run_one, jobs, chunksize=1))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    (out / "config.json").write_text(json.dumps({"honest_error_bound": honest_error, "seeds": SEEDS,
                                                 "attackers": ATTACKERS, "defenses": defenses(honest_error)},
                                                indent=2))
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
