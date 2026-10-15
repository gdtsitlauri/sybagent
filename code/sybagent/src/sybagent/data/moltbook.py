"""Loader for agent-to-agent interactions (e.g. Moltbook archive exports) as CSV (columns: source, target[, timestamp, kind])."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

REQUIRED = {"source", "target"}


def load_interactions(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def load_interaction_graph(path: str | Path, kind: str | None = None) -> nx.DiGraph:
    df = load_interactions(path)
    if kind is not None and "kind" in df.columns:
        df = df[df["kind"] == kind]
    g = nx.DiGraph()
    for row in df.itertuples(index=False):
        u, v = str(row.source), str(row.target)
        if u == v:
            continue
        w = g[u][v]["weight"] + 1 if g.has_edge(u, v) else 1
        g.add_edge(u, v, weight=w)
    return g
