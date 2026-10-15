"""Loader for ERC-8004 reputation-registry feedback exported to CSV (columns: rater, agent, score[, chain, block])."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd

REQUIRED = {"rater", "agent", "score"}


def load_feedback(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return df


def load_feedback_graph(path: str | Path, chain: str | None = None) -> nx.DiGraph:
    df = load_feedback(path)
    if chain is not None and "chain" in df.columns:
        df = df[df["chain"] == chain]
    g = nx.DiGraph()
    for row in df.itertuples(index=False):
        rater, agent, score = str(row.rater), str(row.agent), float(row.score)
        if g.has_edge(rater, agent):
            g[rater][agent]["scores"].append(score)
        else:
            g.add_edge(rater, agent, scores=[score])
    for _, _, d in g.edges(data=True):
        d["weight"] = sum(d["scores"]) / len(d["scores"])
    return g
