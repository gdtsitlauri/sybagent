"""Structure of the Moltbook agent interaction graph (commenter <-> post author), same metrics as for ERC-8004.

    python scripts/analyze_moltbook.py data/raw/moltbook/extracted
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from analyze_erc8004 import spectral_gap  # noqa: E402


def _read(files: list[Path], cols: list[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        schema = pq.read_schema(f)
        use = [c for c in cols if c in schema.names]
        if len(use) == len(cols):
            parts.append(pq.read_table(f, columns=use).to_pandas())
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "moltbook" / "extracted"
    files = sorted(root.rglob("*.parquet"))
    post_files = [f for f in files if "comment" not in str(f).lower() and "post" in str(f).lower()]
    comment_files = [f for f in files if "comment" in str(f).lower()]
    posts = _read(post_files, ["id", "agent_id"]).drop_duplicates("id")
    comments = _read(comment_files, ["post_id", "agent_id"])
    author = dict(zip(posts["id"], posts["agent_id"]))
    comments["post_author"] = comments["post_id"].map(author)
    inter = comments.dropna(subset=["post_author", "agent_id"])
    inter = inter[inter["agent_id"] != inter["post_author"]]

    g = nx.Graph()
    g.add_edges_from(zip(inter["agent_id"], inter["post_author"]))
    comps = sorted(nx.connected_components(g), key=len, reverse=True)
    giant = g.subgraph(comps[0]).copy()
    gap = spectral_gap(giant)
    n_giant = giant.number_of_nodes()
    per_agent = pd.concat([inter["agent_id"], inter["post_author"]]).value_counts()
    partners = pd.Series(dict(g.degree()))
    row = {
        "posts": len(posts),
        "comments": len(comments),
        "interactions_non_self": len(inter),
        "agents_in_graph": g.number_of_nodes(),
        "edges": g.number_of_edges(),
        "top1pct_agents_interaction_share": per_agent.iloc[: max(1, len(per_agent) // 100)].sum() / per_agent.sum(),
        "agents_single_partner_frac": float((partners == 1).mean()),
        "components": len(comps),
        "giant_component_frac": n_giant / g.number_of_nodes(),
        "giant_spectral_gap": gap,
        "giant_mixing_time_est": float(np.log(n_giant) / gap) if gap > 0 else float("inf"),
        "fast_mixing_reference_logn": float(np.log(n_giant)),
    }
    s = pd.Series(row, name="moltbook")
    print(s.to_string())
    (ROOT / "tables").mkdir(exist_ok=True)
    s.to_frame().to_csv(ROOT / "tables" / "moltbook_structure.csv")
    (ROOT / "results" / "8_real_data").mkdir(parents=True, exist_ok=True)
    s.to_frame().to_csv(ROOT / "results" / "8_real_data" / "moltbook_structure.csv")
    (ROOT / "tables" / "moltbook_structure.md").write_text(s.to_frame().to_markdown(floatfmt=".4g"), encoding="utf-8")


if __name__ == "__main__":
    main()
