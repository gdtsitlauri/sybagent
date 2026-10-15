"""Plot a single run or a sweep summary.

    python scripts/plot_results.py run results/default/run.csv --out results/default/run.png
    python scripts/plot_results.py sweep results/sweep_defenses/summary.csv --x adversary.n_sybils \
        --y final_target_sybil_share --hue defense.kind --out results/sweep_defenses/defenses.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sybagent.plotting import plot_run, plot_sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("csv")
    r.add_argument("--out", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("csv")
    s.add_argument("--x", required=True)
    s.add_argument("--y", required=True)
    s.add_argument("--hue")
    s.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.mode == "run":
        print(plot_run(args.csv, args.out))
    else:
        print(plot_sweep(args.csv, args.x, args.y, args.hue, args.out))


if __name__ == "__main__":
    main()
