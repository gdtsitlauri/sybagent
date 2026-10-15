"""Run one config or a sweep.

    python scripts/run_experiment.py --config configs/default.yaml
    python scripts/run_experiment.py --sweep configs/sweep_attack.yaml --out results/sweep_attack
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sybagent import Simulation, load_config  # noqa: E402
from sybagent.experiments import run_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config")
    g.add_argument("--sweep")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.config:
        cfg = load_config(args.config)
        out = Path(args.out or cfg.output_dir)
        sim = Simulation(cfg)
        sim.run()
        print(sim.save(out / "run.csv"))
    else:
        out = Path(args.out or "results/sweep")
        summary = run_sweep(args.sweep, out, workers=args.workers)
        print(f"{len(summary)} runs -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
