"""Eclipse, Sybil-exposure and service metrics (computed with ground-truth labels)."""

from __future__ import annotations

import networkx as nx
import numpy as np

from sybagent.context import SimContext


def sybil_share(neighbors: list[int], sybil_ids: set[int]) -> float:
    if not neighbors:
        return 0.0
    return sum(n in sybil_ids for n in neighbors) / len(neighbors)


def honest_neighbor_count(neighbors: list[int], sybil_ids: set[int]) -> int:
    return sum(n not in sybil_ids for n in neighbors)


def honest_component_fraction(ctx: SimContext) -> float:
    """Largest weakly connected component of the honest-only peer graph, as a fraction of honest agents."""
    honest = set(ctx.honest_ids)
    if not honest:
        return 0.0
    g = ctx.peer_graph().subgraph(honest)
    if g.number_of_nodes() == 0:
        return 0.0
    return max(len(c) for c in nx.weakly_connected_components(g)) / len(honest)


def round_metrics(ctx: SimContext, targets: list[int], qualities: list[float], eclipse_threshold: float = 1.0) -> dict:
    sy = ctx.sybil_ids
    honest = [ctx.agents[i] for i in ctx.honest_ids]
    shares = np.array([sybil_share(a.neighbors, sy) for a in honest]) if honest else np.array([0.0])
    hon_counts = np.array([honest_neighbor_count(a.neighbors, sy) for a in honest]) if honest else np.array([0])
    t_shares = [sybil_share(ctx.agents[t].neighbors, sy) for t in targets if t in ctx.agents]
    t_honest = [honest_neighbor_count(ctx.agents[t].neighbors, sy) for t in targets if t in ctx.agents]
    eclipsed = [s >= eclipse_threshold and len(ctx.agents[t].neighbors) > 0
                for s, t in zip(t_shares, targets)]
    pinned_honest = [honest_neighbor_count(list(a.pinned), sy) for a in honest if a.pinned]
    no_honest_anchor = [c == 0 for c in pinned_honest]
    return {
        "round": ctx.round,
        "target_sybil_share": float(np.mean(t_shares)) if t_shares else 0.0,
        "target_min_honest_neighbors": int(min(t_honest)) if t_honest else 0,
        "targets_eclipsed_frac": float(np.mean(eclipsed)) if eclipsed else 0.0,
        "honest_mean_sybil_share": float(shares.mean()),
        "honest_exposed_frac": float((shares > 0).mean()),
        "honest_eclipsed_frac": float((shares >= eclipse_threshold).mean()),
        "honest_min_honest_neighbors": int(hon_counts.min()),
        "honest_component_frac": honest_component_fraction(ctx),
        "honest_no_honest_anchor_frac": float(np.mean(no_honest_anchor)) if no_honest_anchor else float("nan"),
        "mean_quality": float(np.mean(qualities)) if qualities else float("nan"),
        "n_sybils": len(sy),
        "llm_calls": ctx.llm_calls,
        "audit_calls": ctx.audit_calls,
    }


def time_to_eclipse(rows: list[dict], key: str = "targets_eclipsed_frac", level: float = 1.0) -> int | None:
    for r in rows:
        if r[key] >= level:
            return r["round"]
    return None
