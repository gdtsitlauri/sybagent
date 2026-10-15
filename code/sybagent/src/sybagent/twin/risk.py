"""Eclipse risk and what-if analysis on the twin state.

Discovery model. A client looking for a service category (feedback tag) is shown the top-k agents of that tag by
number of distinct raters ("reputation discovery"). With the defense, it also keeps `a` anchors sampled uniformly from
the identity registry.

Attack costs at time t
  * reputation discovery, full eclipse: k Sybil agents, each rated by s_1 + 1 fake raters, where s_1 is the best
    honest agent's rater count in the tag. Fake raters are plain addresses (ERC-8004 does not require raters to be
    registered agents) and each rates all k Sybils, so the adversary needs s_1 + 1 rater addresses and k agents.
  * with anchors: the eclipse also needs every anchor to be a Sybil. With N registered agents and B Sybil agents,
    P(no honest anchor) <= (B / (N + B - a))^a (Theorem 1), so reaching probability eps needs
    B >= eps^(1/a) (N - a) / (1 - eps^(1/a)) registered identities.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from sybagent.twin.state import DAY, Snapshot, TwinState


def tag_rankings(snap: Snapshot, min_agents: int = 1) -> dict[str, pd.Series]:
    """tag -> distinct-rater counts per agent, sorted descending."""
    fb = snap.feedback[snap.feedback["tag1"].fillna("") != ""]
    counts = fb.groupby(["tag1", "agent"])["rater"].nunique()
    out = {}
    for tag, s in counts.groupby(level=0):
        s = s.droplevel(0).sort_values(ascending=False, kind="stable")
        if len(s) >= min_agents:
            out[str(tag)] = s
    return out


def reputation_eclipse_cost(ranking: pd.Series, k: int) -> dict:
    s = ranking.to_numpy()
    s1 = int(s[0]) if len(s) else 0
    sk = int(s[k - 1]) if len(s) >= k else 0
    return {"agents": len(s), "top1_raters": s1, "kth_raters": sk,
            "full_eclipse_raters": s1 + 1, "full_eclipse_sybil_agents": k,
            "full_eclipse_feedback_tx": k * (s1 + 1), "one_slot_raters": sk + 1}


def anchors_identities_needed(n_registered: int, a: int, eps: float) -> float:
    """Smallest B with (B / (N + B - a))^a >= eps."""
    r = eps ** (1.0 / a)
    return r * (n_registered - a) / (1 - r)


def anchors_no_honest_exact(n_honest: int, b: int, a: int) -> float:
    """Exact hypergeometric P(all a anchors are Sybil) when drawing from N + B identities (requester excluded)."""
    n = n_honest + b
    return comb(b, a) / comb(n, a) if b >= a else 0.0


def risk_table(state: TwinState, t: float, k: int = 5, a: int = 3, eps: float = 0.5,
               min_agents: int = 5) -> pd.DataFrame:
    snap = state.at(t)
    rows = []
    b_needed = anchors_identities_needed(snap.n_registered, a, eps)
    for tag, ranking in tag_rankings(snap, min_agents).items():
        c = reputation_eclipse_cost(ranking, k)
        rows.append({"t": t, "tag": tag, **c, "registered": snap.n_registered,
                     "reputation_identities": c["full_eclipse_raters"] + k,
                     "anchors_identities": b_needed + c["full_eclipse_raters"] + k})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ what-if: apply an attack to a copy of the state
def apply_attack(snap: Snapshot, tag: str, n_sybil_agents: int, n_fake_raters: int, t: float) -> Snapshot:
    """Add Sybil agents (registered) and fake raters that each rate every Sybil agent in `tag`."""
    sy = [f"sybil{i}" for i in range(n_sybil_agents)]
    fake = pd.DataFrame([{"chain": "whatif", "block": -1, "tx": "", "rater": f"fake{j}", "agent": s,
                          "feedback_index": 0, "score": 100.0, "tag1": tag, "tag2": "", "revoked": False, "ts": t}
                         for s in sy for j in range(n_fake_raters)])
    fb = pd.concat([snap.feedback, fake], ignore_index=True) if len(fake) else snap.feedback
    return Snapshot(t=t, feedback=fb, registered=np.concatenate([snap.registered, np.array(sy)]))


def topk_sybil_share(snap: Snapshot, tag: str, k: int) -> float:
    r = tag_rankings(snap).get(tag)
    if r is None:
        return 0.0
    top = list(r.index[:k])
    return sum(str(x).startswith("sybil") for x in top) / max(1, len(top))


def anchors_monte_carlo(snap: Snapshot, a: int, n_draws: int, rng: np.random.Generator) -> float:
    reg = snap.registered
    sy = np.array([str(x).startswith("sybil") for x in reg])
    hits = 0
    for _ in range(n_draws):
        idx = rng.choice(len(reg), size=a, replace=False)
        hits += bool(sy[idx].all())
    return hits / n_draws


# ------------------------------------------------------------------ backtest of the risk score
def takeover_events(state: TwinState, t: float, horizon: float, k: int, min_agents: int,
                    fresh_share: float = 0.5) -> pd.DataFrame:
    """For each tag ranked at t: did an agent enter the top-k by t + horizon, mostly through raters that were new
    (first feedback ever inside the window)? Such boosted entries are the observable footprint of reputation
    manipulation (or of genuinely new demand)."""
    before = tag_rankings(state.at(t), min_agents)
    after = tag_rankings(state.at(t + horizon), 1)
    win = state.window(t, t + horizon)
    first_seen = state.feedback.groupby("rater")["ts"].min()
    fresh = set(first_seen[(first_seen >= t) & (first_seen < t + horizon)].index)
    recent = state.window(t - 28 * DAY, t).groupby("tag1").size()
    rows = []
    for tag, rk in before.items():
        top_before = set(rk.index[:k])
        rk_after = after.get(tag, rk)
        w = win[win["tag1"] == tag]
        boosted = 0
        for agent in rk_after.index[:k]:
            if agent in top_before:
                continue
            raters = set(w.loc[w["agent"] == agent, "rater"])
            if raters and len(raters & fresh) / len(raters) >= fresh_share:
                boosted += 1
        rows.append({"t": t, "tag": tag, "boosted_entries": boosted, "takeover": int(boosted > 0),
                     "activity_28d": int(recent.get(tag, 0))})
    return pd.DataFrame(rows)
