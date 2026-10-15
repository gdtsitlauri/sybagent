"""Figures from run CSVs and sweep summaries."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_run(csv_path: str | Path, out_path: str | Path) -> Path:
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(df["round"], df["target_sybil_share"], label="target: Sybil share of peers")
    ax.plot(df["round"], df["honest_mean_sybil_share"], label="all honest: mean Sybil share")
    ax.set_xlabel("round")
    ax.set_ylabel("share")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_sweep(summary_csv: str | Path, x: str, y: str, hue: str | None, out_path: str | Path) -> Path:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    groups = df.groupby(hue) if hue else [(None, df)]
    for name, g in groups:
        agg = g.groupby(x)[y].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg[x], agg["mean"], yerr=agg["std"], marker="o", capsize=3,
                    label=str(name) if name is not None else None)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if hue:
        ax.legend(title=hue, frameon=False)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
