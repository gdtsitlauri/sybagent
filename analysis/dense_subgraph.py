"""Study 13: dense-subgraph detection of coordinated raters in ERC-8004 feedback graphs.

Graph: bipartite, rater address -- rated agent id, one edge per distinct (rater, agent) pair.
Methods
  * density     Charikar's greedy peeling for max average degree |E(S)|/|S| (a 2-approximation);
  * weighted    the same with Fraudar's column weights 1/log(deg(agent)+5) (Hooi et al., KDD 2016), which resists
                camouflage edges towards popular agents;
  * exact       Goldberg's max-flow formulation, used to check the greedy result on the Ethereum graph.
Blocks are extracted one after another (edges of a found block are removed). Each block's score is compared with the
best block of degree-preserving random graphs (null model), giving a z-score.
An injection test plants synthetic Sybil blocks (with and without camouflage) in the real graph and measures
precision and recall of their nodes in the top blocks.

    python analysis/dense_subgraph.py
"""

from __future__ import annotations

import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "erc8004"
OUT = ROOT / "results" / "13_dense_subgraph"
K_BLOCKS = 10
N_NULL = 20
INJECT_SEEDS = 5


def load_edges(chain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(DATA / f"{chain}_feedback.csv", dtype={"agent": str, "rater": str})
    d = d[~d["revoked"].astype(bool)]
    pairs = d.groupby(["rater", "agent"]).size().rename("n").reset_index()
    return d, pairs


def peel(edges: list[tuple[str, str]], weighted: bool) -> tuple[set[str], float]:
    """Greedy peeling. Nodes are 'r:<addr>' and 'a:<id>'. Returns the best node set and its score."""
    adj: dict[str, dict[str, float]] = defaultdict(dict)
    col_deg: dict[str, int] = defaultdict(int)
    for r, a in edges:
        col_deg[a] += 1
    for r, a in edges:
        w = 1.0 / math.log(col_deg[a] + 5) if weighted else 1.0
        adj["r:" + r]["a:" + a] = w
        adj["a:" + a]["r:" + r] = w
    load = {v: sum(nb.values()) for v, nb in adj.items()}
    total = sum(load.values()) / 2
    alive = set(adj)
    heap = [(l, v) for v, l in load.items()]
    heapq.heapify(heap)
    best, best_k, order = total / max(1, len(alive)), 0, []
    while alive:
        l, v = heapq.heappop(heap)
        if v not in alive or abs(l - load[v]) > 1e-12:
            continue
        alive.discard(v)
        order.append(v)
        total -= load[v]
        for u, w in adj[v].items():
            if u in alive:
                load[u] -= w
                heapq.heappush(heap, (load[u], u))
        if alive:
            score = total / len(alive)
            if score > best:
                best, best_k = score, len(order)
    removed = set(order[:best_k])
    return set(adj) - removed, best


def blocks(edges: list[tuple[str, str]], weighted: bool, k: int = K_BLOCKS) -> list[dict]:
    rest = list(edges)
    out = []
    for i in range(k):
        if not rest:
            break
        nodes, score = peel(rest, weighted)
        inside = [(r, a) for r, a in rest if "r:" + r in nodes and "a:" + a in nodes]
        if not inside:
            break
        out.append({"block": i, "score": score, "nodes": nodes, "edges": inside})
        keep = set(inside)
        rest = [e for e in rest if e not in keep]
    return out


def exact_density(edges: list[tuple[str, str]]) -> float:
    """Goldberg's max-flow test for max |E(S)|/|S| (binary search on the density g)."""
    g = nx.Graph()
    g.add_edges_from(("r:" + r, "a:" + a) for r, a in edges)
    m = g.number_of_edges()
    deg = dict(g.degree())
    lo, hi = 0.0, float(max(deg.values()))
    nodes = list(g)
    while hi - lo > 1.0 / (len(nodes) * (len(nodes) - 1) + 1) and hi - lo > 1e-4:
        mid = (lo + hi) / 2
        f = nx.DiGraph()
        for v in nodes:
            f.add_edge("_s", v, capacity=m)
            f.add_edge(v, "_t", capacity=m + 2 * mid - deg[v])
        for u, v in g.edges():
            f.add_edge(u, v, capacity=1.0)
            f.add_edge(v, u, capacity=1.0)
        _, (s_side, _) = nx.minimum_cut(f, "_s", "_t")
        if len(s_side) > 1:
            lo = mid
        else:
            hi = mid
    return lo


def null_edges(edges: list[tuple[str, str]], rng: np.random.Generator) -> list[tuple[str, str]]:
    """Degree-preserving random bipartite graph (stub matching, duplicate pairs dropped)."""
    rs = [r for r, _ in edges]
    as_ = [a for _, a in edges]
    rng.shuffle(as_)
    return list(dict.fromkeys(zip(rs, as_)))


def describe(b: dict, d: pd.DataFrame, n_agents: int, n_feedback: int) -> dict:
    raters = {v[2:] for v in b["nodes"] if v.startswith("r:")}
    agents = {v[2:] for v in b["nodes"] if v.startswith("a:")}
    sub = d[d["rater"].isin(raters) & d["agent"].isin(agents)]
    return {"block": b["block"], "score": b["score"], "raters": len(raters), "agents": len(agents),
            "pairs": len(b["edges"]), "pair_density": len(b["edges"]) / max(1, len(raters) * len(agents)),
            "feedback_events": len(sub), "feedback_share": len(sub) / n_feedback,
            "agents_share": len(agents) / n_agents,
            "top_tag": sub["tag1"].value_counts().index[0] if len(sub) else "",
            "top_tag_share": float(sub["tag1"].value_counts(normalize=True).iloc[0]) if len(sub) else float("nan"),
            "block_span": int(sub["block"].max() - sub["block"].min()) if len(sub) else 0}


def inject(edges, rng, n_r=40, n_a=40, density=0.2, camo=0):
    """Plant a Sybil block: n_r raters x n_a agents with the given pair density, plus `camo` camouflage edges per
    Sybil rater towards popular real agents (chosen proportionally to their degree)."""
    agents = [a for _, a in edges]
    sr = [f"syb_r{i}" for i in range(n_r)]
    sa = [f"syb_a{i}" for i in range(n_a)]
    planted = [(r, a) for r in sr for a in sa if rng.random() < density]
    camo_e = [(r, agents[int(rng.integers(len(agents)))]) for r in sr for _ in range(camo)]
    return list(dict.fromkeys(edges + planted + camo_e)), {"r:" + r for r in sr} | {"a:" + a for a in sa}


def injection_test(edges, chain: str) -> list[dict]:
    rows = []
    for density in (0.05, 0.1, 0.2, 0.5):
        for camo in (0, 10):
            for seed in range(INJECT_SEEDS):
                rng = np.random.default_rng([seed, int(density * 100), camo])
                e2, truth = inject(edges, rng, density=density, camo=camo)
                for weighted in (False, True):
                    best = (0.0, 0.0, 0.0, -1)     # f1, precision, recall, rank of the block that matches best
                    for b in blocks(e2, weighted, k=K_BLOCKS):
                        tp = len(b["nodes"] & truth)
                        if tp:
                            p, r = tp / len(b["nodes"]), tp / len(truth)
                            f1 = 2 * p * r / (p + r)
                            if f1 > best[0]:
                                best = (f1, p, r, b["block"])
                    rows.append({"chain": chain, "density": density, "camouflage": camo, "seed": seed,
                                 "method": "weighted" if weighted else "density", "f1": best[0],
                                 "precision": best[1], "recall": best[2], "rank": best[3]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_blocks, nulls, summary, inj = [], [], {}, []
    for chain in ("ethereum", "base"):
        d, pairs = load_edges(chain)
        edges = list(zip(pairs["rater"], pairs["agent"]))
        n_agents, n_feedback = d["agent"].nunique(), len(d)
        rng = np.random.default_rng(0)
        info = {"feedback": n_feedback, "pairs": len(edges), "raters": d["rater"].nunique(), "agents": n_agents}
        for weighted in (False, True):
            method = "weighted" if weighted else "density"
            bl = blocks(edges, weighted)
            null_scores = np.full((N_NULL, K_BLOCKS), np.nan)    # score of the k-th block of each null graph
            for j in range(N_NULL):
                for nb in blocks(null_edges(edges, rng), weighted):
                    null_scores[j, nb["block"]] = nb["score"]
            nulls += [{"chain": chain, "method": method, "null": j, "block": k, "score": null_scores[j, k]}
                      for j in range(N_NULL) for k in range(K_BLOCKS)]
            top_mu, top_sd = float(np.mean(null_scores[:, 0])), float(np.std(null_scores[:, 0], ddof=1))
            agents_in = set()
            for b in bl:
                row = describe(b, d, n_agents, n_feedback)
                k = b["block"]
                col = null_scores[:, k][~np.isnan(null_scores[:, k])]
                mu = float(col.mean()) if len(col) else float("nan")
                sd = float(col.std(ddof=1)) if len(col) > 1 else float("nan")
                row.update(chain=chain, method=method, null_mean=mu, null_sd=sd,
                           z=(b["score"] - mu) / sd if sd and sd > 0 else float("nan"),
                           z_vs_null_top=(b["score"] - top_mu) / top_sd if top_sd > 0 else float("nan"))
                all_blocks.append(row)
                if row["z"] > 3:
                    agents_in |= {v[2:] for v in b["nodes"] if v.startswith("a:")}
            info[f"{method}_significant_blocks"] = int(sum(r["z"] > 3 for r in all_blocks
                                                           if r["chain"] == chain and r["method"] == method))
            info[f"{method}_agents_in_significant_blocks"] = len(agents_in)
            info[f"{method}_agents_in_significant_blocks_frac"] = len(agents_in) / n_agents
        if chain == "ethereum":
            info["exact_max_density"] = exact_density(edges)
            info["greedy_max_density"] = peel(edges, False)[1]
        summary[chain] = info
        inj += injection_test(edges, chain)
        print(chain, json.dumps(info, indent=1, default=float), flush=True)
    pd.DataFrame(all_blocks).to_csv(OUT / "blocks.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "null_scores.csv", index=False)
    pd.DataFrame(inj).to_csv(OUT / "injection.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
