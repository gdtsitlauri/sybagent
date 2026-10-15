import networkx as nx

from sybagent.defenses.sumup import accepted_voters
from sybagent.defenses.sybilrank import sybilrank_scores


def _two_regions():
    """Honest clique 0..5 and Sybil clique 10..15 joined by a single attack edge (5, 10)."""
    g = nx.Graph()
    g.add_edges_from(nx.complete_graph(range(6)).edges)
    g.add_edges_from(nx.complete_graph(range(10, 16)).edges)
    g.add_edge(5, 10)
    return g


def test_sybilrank_ranks_honest_above_sybil():
    g = _two_regions()
    adj = {v: list(g.neighbors(v)) for v in g.nodes}
    s = sybilrank_scores(adj, seeds=[0, 1])
    honest = [s[v] for v in range(6)]
    sybil = [s[v] for v in range(10, 16)]
    assert min(honest) > max(sybil)


def test_sumup_limits_votes_through_attack_edge():
    g = _two_regions()
    ok = accepted_voters(g, collector=0, voters=set(range(1, 6)) | set(range(10, 16)), near_capacity=4)
    assert len(ok & set(range(10, 16))) <= 1
    assert len(ok & set(range(1, 6))) >= 4
