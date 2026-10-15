"""Structure of real ERC-8004 feedback graphs vs. the assumptions of graph-based Sybil defenses.

    python scripts/analyze_erc8004.py data/raw/erc8004/ethereum_feedback.csv data/raw/erc8004/base_feedback.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]


def spectral_gap(g: nx.Graph) -> float:
    """1 - |lambda_2| of the lazy random walk on the largest component (larger = faster mixing)."""
    if g.number_of_nodes() < 3:
        return float("nan")
    a = nx.to_scipy_sparse_array(g, format="csr", dtype=float)
    deg = np.asarray(a.sum(axis=1)).ravel()
    d_inv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    s = d_inv_sqrt @ a @ d_inv_sqrt                     # symmetric normalized adjacency
    vals = eigsh(s, k=3, which="LA", return_eigenvectors=False, tol=1e-6, maxiter=20000)
    vals = np.sort(vals)[::-1]
    lam2_lazy = 0.5 * (1.0 + vals[1])                   # lazy walk removes bipartite -1 eigenvalue
    return float(1.0 - lam2_lazy)


def analyze(path: Path) -> dict:
    d = pd.read_csv(path)
    d = d[~d["revoked"].astype(bool)]
    chain = d["chain"].iloc[0] if len(d) else path.stem
    per_rater = d.groupby("rater").size().sort_values(ascending=False)
    distinct_raters = d.groupby("agent")["rater"].nunique()
    agents_per_rater = d.groupby("rater")["agent"].nunique()

    g = nx.Graph()
    g.add_edges_from(("r:" + r, "a:" + str(a)) for r, a in zip(d["rater"], d["agent"]))
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    giant = g.subgraph(comps[0]).copy() if comps else nx.Graph()
    gap = spectral_gap(giant)
    n_giant = giant.number_of_nodes()
    mix = (np.log(max(n_giant, 2)) / gap) if gap and gap > 0 else float("inf")

    return {
        "chain": chain,
        "feedback": len(d),
        "raters": d["rater"].nunique(),
        "agents_rated": d["agent"].nunique(),
        "top1_rater_share": per_rater.iloc[:1].sum() / len(d),
        "top10_rater_share": per_rater.iloc[:10].sum() / len(d),
        "agents_single_rater_frac": float((distinct_raters == 1).mean()),
        "agents_le2_raters_frac": float((distinct_raters <= 2).mean()),
        "raters_single_agent_frac": float((agents_per_rater == 1).mean()),
        "components": len(comps),
        "giant_component_frac": n_giant / g.number_of_nodes() if g.number_of_nodes() else 0.0,
        "giant_spectral_gap": gap,
        "giant_mixing_time_est": mix,
        "fast_mixing_reference_logn": float(np.log(max(n_giant, 2))),
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted((ROOT / "data" / "raw" / "erc8004").glob("*_feedback.csv"))
    rows = [analyze(p) for p in paths if p.exists() and p.stat().st_size > 100]
    df = pd.DataFrame(rows).set_index("chain").T
    print(df.to_string())
    (ROOT / "tables").mkdir(exist_ok=True)
    df.to_csv(ROOT / "tables" / "erc8004_structure.csv")
    (ROOT / "results" / "8_real_data").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "results" / "8_real_data" / "erc8004_structure.csv")
    (ROOT / "tables" / "erc8004_structure.md").write_text(df.to_markdown(floatfmt=".4g"), encoding="utf-8")


if __name__ == "__main__":
    main()
