"""Parameter sweeps: base config + grid of dotted-key overrides, repeated over seeds."""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from sybagent.config import from_dict, merge
from sybagent.metrics import time_to_eclipse
from sybagent.simulation import Simulation


def _dotted_to_nested(key: str, value: Any) -> dict:
    out: dict = {}
    cur = out
    parts = key.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def expand_sweep(spec: dict) -> list[dict]:
    """spec = {"base": {...}, "grid": {"a.b": [..], ...}, "seeds": [..]} -> list of full config dicts."""
    base = spec.get("base", {})
    grid = spec.get("grid", {})
    seeds = spec.get("seeds", [base.get("seed", 0)])
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys))) if keys else [()]
    runs = []
    for combo in combos:
        cfg = dict(base)
        for k, v in zip(keys, combo):
            cfg = merge(cfg, _dotted_to_nested(k, v))
        for s in seeds:
            runs.append(merge(cfg, {"seed": s}))
    return runs


def _lookup(cfg_dict: dict, dotted: str) -> Any:
    node: Any = cfg_dict
    for p in dotted.split("."):
        node = node[p]
    return node


def _run_one(args: tuple[int, dict, list[str], str]) -> dict:
    i, cfg_dict, grid_keys, out_dir = args
    out = Path(out_dir)
    cfg = from_dict(cfg_dict)
    sim = Simulation(cfg)
    df = sim.run()
    sim.save(out / "runs" / f"run_{i:04d}.csv")
    (out / "runs" / f"run_{i:04d}.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    row: dict[str, Any] = {"run": i, "seed": cfg.seed}
    for k in grid_keys:
        row[k] = _lookup(cfg_dict, k)
    last = df.iloc[-1].to_dict()
    row.update({f"final_{k}": v for k, v in last.items() if k != "round"})
    tail = df.tail(min(20, len(df))).mean(numeric_only=True)
    row.update({f"tail_{k}": float(v) for k, v in tail.items() if k != "round"})
    row["time_to_eclipse"] = time_to_eclipse(df.to_dict("records"))
    return row


def run_sweep(spec_path: str | Path, out_dir: str | Path, workers: int = 1) -> pd.DataFrame:
    with open(spec_path, "r", encoding="utf-8-sig") as fh:
        spec = yaml.safe_load(fh)
    out_dir = Path(out_dir)
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    grid_keys = list(spec.get("grid", {}))
    jobs = [(i, c, grid_keys, str(out_dir)) for i, c in enumerate(expand_sweep(spec))]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            summary = list(ex.map(_run_one, jobs, chunksize=1))
    else:
        summary = [_run_one(j) for j in jobs]
    out = pd.DataFrame(summary).sort_values("run")
    out.to_csv(out_dir / "summary.csv", index=False)
    return out
