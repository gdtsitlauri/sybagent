"""Re-run stored runs of every simulation study with the current code and compare with results/.

Writes results/repro_check.json. Columns that exist in both the stored and the new run must match exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "sybagent" / "src"))
from sybagent.config import from_dict  # noqa: E402
from sybagent.simulation import Simulation  # noqa: E402

CHECKS = {"1_attack": [5, 300], "2_defenses": [0, 59, 299], "3_anchor_bound": [41],
          "4_adaptive_attacks": [7], "5_second_opinions": [37, 230],
          "5b_robust_second_opinions/baselines": [13], "5b_robust_second_opinions/main": [50, 119],
          "12_pareto": [77, 512]}


def main() -> None:
    out = []
    for study, idx in CHECKS.items():
        for i in idx:
            f = ROOT / "results" / study / "runs" / f"run_{i:04d}.csv"
            if not f.exists():
                continue
            cfg = json.loads(f.with_suffix(".json").read_text())
            old, new = pd.read_csv(f), Simulation(from_dict(cfg)).run()
            cols = [c for c in old.columns if c in new.columns]
            same = all(((old[c] - new[c]).abs().fillna(0) < 1e-9).all() for c in cols)
            out.append({"study": study, "run": i, "columns_compared": len(cols), "identical": bool(same)})
            print(study, i, "identical" if same else "DIFFERENT")
    # study 11: runs are named by configuration; RL evaluation reloads the saved Q-table
    sys.path.insert(0, str(ROOT / "code" / "sybagent" / "scripts"))
    import run_strategic as rs  # noqa: E402
    s11 = ROOT / "results" / "11_strategic_attacker"
    for b, u, aa, rule, att, seed in [(50, 0.3, True, "majority", "lie1.0", 0), (200, 0.6, False, "unanimous", "budget", 2),
                                      (200, 0.3, False, "majority", "rl", 1)]:
        tag = f"B{b}_u{u}_aa{int(aa)}_{rule}"
        f = s11 / "runs" / f"{tag}_{att}_s{seed}.csv"
        if not f.exists():
            continue
        if att == "rl":
            g = json.loads((s11 / "q" / f"{tag}_policy.json").read_text())["gamma"]
            adv = {"policy": "rl", "policy_q": str(s11 / "q" / f"{tag}_g{g}.json")}
        else:
            adv = rs.FIXED[att]
        old, new = pd.read_csv(f), Simulation(from_dict(rs.config(b, u, aa, rule, seed, adv))).run()
        cols = [c for c in old.columns if c in new.columns]
        same = all(((old[c] - new[c]).abs().fillna(0) < 1e-9).all() for c in cols)
        out.append({"study": "11_strategic_attacker", "run": f.stem, "columns_compared": len(cols), "identical": bool(same)})
        print("11_strategic_attacker", f.stem, "identical" if same else "DIFFERENT")
    # study 20: re-run summary rows of the shared-defense sweep (runs are stored as summaries only)
    import run_shared as sh  # noqa: E402
    main_csv = ROOT / "results" / "20_shared_defense" / "main.csv"
    if main_csv.exists():
        m = pd.read_csv(main_csv)
        for b, share, att, seed in [(50, "alerts+committee", "lie1.0", 0), (100, "committee", "budget", 1)]:
            row = m[(m["n_sybils"] == b) & (m["share"] == share) & (m["attacker"] == att) &
                    (m["false_reports"] == 1) & (m["seed"] == seed)].iloc[0]
            new = sh.summarize(Simulation(from_dict(sh.config(b, seed, sh.ATTACKERS[att], {"share": share}, 1))).run())
            keys = ["wrong_used", "global_blocks_sybil", "global_blocks_honest", "queries_per_task"]
            same = all(abs(float(row[k]) - float(new[k])) < 1e-9 for k in keys)
            tag = f"B{b}_{share}_{att}_s{seed}"
            out.append({"study": "20_shared_defense", "run": tag, "columns_compared": len(keys), "identical": same})
            print("20_shared_defense", tag, "identical" if same else "DIFFERENT")
    # study 22: verifiable answers
    import run_verify as rv  # noqa: E402
    v_csv = ROOT / "results" / "22_verifiable" / "summary.csv"
    if v_csv.exists():
        v = pd.read_csv(v_csv)
        for b, vp, mode, att, seed in [(200, 0.05, "ban+rep", "sleeper", 1)]:
            row = v[(v["n_sybils"] == b) & (v["verify_prob"] == vp) & (v["mode"] == mode) & (v["attacker"] == att)
                    & (v["seed"] == seed)].iloc[0]
            new = rv.summarize(Simulation(from_dict(sh.config(b, seed, rv.ATTACKERS[att], rv.params(vp, mode), 0))).run())
            keys = ["wrong_used", "global_blocks_sybil", "global_blocks_honest", "verifications_per_task"]
            same = all(abs(float(row[k]) - float(new[k])) < 1e-9 for k in keys)
            tag = f"B{b}_v{vp}_{mode}_{att}_s{seed}"
            out.append({"study": "22_verifiable", "run": tag, "columns_compared": len(keys), "identical": same})
            print("22_verifiable", tag, "identical" if same else "DIFFERENT")
    (ROOT / "results" / "repro_check.json").write_text(json.dumps(out, indent=2))
    ok = sum(r["identical"] for r in out)
    print(f"{ok}/{len(out)} identical")


if __name__ == "__main__":
    main()
