"""Tests for the extensions: strategic attacker (study 11), on-chain reference sampler (10), dense subgraphs (13),
digital twin risk and what-if (14)."""

import importlib.util
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sybagent.config import from_dict
from sybagent.simulation import Simulation
from sybagent.twin import risk
from sybagent.twin.forecast import auc, holt_forecast
from sybagent.twin.state import Snapshot

HERE = Path(__file__).resolve()


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _small(policy: str, **adv):
    return {"rounds": 6, "network": {"n_honest": 30},
            "adversary": {"n_sybils": 10, "n_targets": 2, "adaptive_cards": True, "coordinated_lies": True,
                          "policy": policy, **adv},
            "selection": {"kind": "reputation"},
            "defense": {"kind": "proposed", "params": {"n_anchors": 2, "audit_prob": 1.0, "audit_flag": 3}}}


def test_fixed_policy_keeps_old_columns_only():
    df = Simulation(from_dict(_small("fixed"))).run()
    assert "strategy_lie_frac" not in df.columns


def test_budget_policy_respects_lie_budget():
    sim = Simulation(from_dict(_small("budget")))
    sim.run()
    assert max(sim.strategy.lies.values(), default=0) <= sim.strategy.budget == 2


def test_rl_policy_trains_and_roundtrips(tmp_path):
    sim = Simulation(from_dict(_small("rl", policy_train=True)))
    sim.run()
    assert np.abs(sim.strategy.q).sum() > 0
    q = tmp_path / "q.json"
    sim.strategy.save(q)
    sim2 = Simulation(from_dict(_small("rl", policy_q=str(q))))
    assert np.allclose(sim2.strategy.q, sim.strategy.q)
    assert not sim2.strategy.train


def test_audit_anchors_can_block_an_anchor():
    cfg = _small("budget", budget_lies=10**6, n_sybils=40)
    cfg["defense"]["params"].update(audit_anchors=True, n_anchors=3, audit_flag=1)
    sim = Simulation(from_dict(cfg))
    sim.run()
    assert sim.results()["blocked_sybil_pairs"].iloc[-1] > 0


def test_reference_draw_distinct_and_in_range():
    mod = _load(HERE.parents[1] / "scripts" / "onchain_sampler.py", "onchain_sampler")
    for seed in range(50):
        out = mod.draw(seed * 7919 + 1, 20, 5, 1)
        assert len(set(out)) == 5 and 1 not in out and all(1 <= x <= 20 for x in out)


def test_grinding_matches_formula_small():
    mod = _load(HERE.parents[1] / "scripts" / "onchain_sampler.py", "onchain_sampler")
    rows = mod.grinding_study(n_honest=10, sybil_counts=(10,), anchors=(2,), rerolls=(0, 3), trials=3000)
    for r in rows:
        assert abs(r["p_no_honest_anchor"] - r["prop6_prediction"]) < 0.05
    assert rows[0]["q_exact"] == pytest.approx(comb(10, 2) / comb(19, 2))


def test_greedy_peeling_finds_planted_biclique():
    mod = _load(HERE.parents[3] / "analysis" / "dense_subgraph.py", "dense_subgraph")
    rng = np.random.default_rng(0)
    edges = [(f"r{rng.integers(200)}", f"a{rng.integers(300)}") for _ in range(400)]
    planted = [(f"R{i}", f"A{j}") for i in range(8) for j in range(8)]
    nodes, score = mod.peel(list(dict.fromkeys(edges + planted)), weighted=False)
    assert {f"r:R{i}" for i in range(8)} <= nodes and score >= 64 / 16
    assert mod.exact_density(planted) == pytest.approx(4.0, abs=1e-3)


def _snap():
    rows = []
    for agent, n in [("1", 4), ("2", 3), ("3", 2), ("4", 1), ("5", 1), ("6", 1)]:
        rows += [{"agent": agent, "rater": f"r{agent}_{j}", "tag1": "t", "ts": 0.0} for j in range(n)]
    return Snapshot(t=1.0, feedback=pd.DataFrame(rows), registered=np.array([str(i) for i in range(1, 101)]))


def test_reputation_cost_is_exactly_sufficient():
    snap = _snap()
    c = risk.reputation_eclipse_cost(risk.tag_rankings(snap)["t"], k=3)
    assert c["full_eclipse_raters"] == 5 and c["one_slot_raters"] == 3
    assert risk.topk_sybil_share(risk.apply_attack(snap, "t", 3, 5, 1.0), "t", 3) == 1.0
    assert risk.topk_sybil_share(risk.apply_attack(snap, "t", 3, 4, 1.0), "t", 3) < 1.0


def test_anchor_identities_needed_inverts_bound():
    from sybagent.defenses.proposed import anchor_eclipse_bound
    b = risk.anchors_identities_needed(1000, 3, 0.5)
    assert anchor_eclipse_bound(1000, int(np.ceil(b)), 3) >= 0.5 > anchor_eclipse_bound(1000, int(b) - 1, 3)


def test_forecast_helpers():
    assert holt_forecast(np.arange(10, dtype=float), 2) == pytest.approx(11.0, abs=0.5)
    assert auc(np.array([0.1, 0.9, 0.8, 0.2]), np.array([0, 1, 1, 0.0])) == 1.0


def test_dp_always_lie_is_optimal():
    mod = _load(HERE.parents[3] / "analysis" / "dp_attacker.py", "dp_attacker")
    for replaceable in (True, False):
        V, _ = mod.make_model(12, 0.27, 0.0, 3, replaceable)
        always = mod.evaluate(12, 0.27, 0.0, 3, replaceable, lambda age, lies: 1)
        assert V(0, 0, 0)[0] == pytest.approx(always, abs=1e-9)


def _shared(mode: str, b: int = 20, fr: int = 1):
    return {"rounds": 8, "network": {"n_honest": 40},
            "adversary": {"n_sybils": b, "n_targets": 2, "adaptive_cards": True, "coordinated_lies": True,
                          "answer_wrong_prob": 1.0, "false_reports": fr},
            "selection": {"kind": "reputation"},
            "defense": {"kind": "shared", "params": {"n_anchors": 3, "audit_prob": 1.0, "audit_flag": 1,
                                                     "share": mode, "committee_size": 9}}}


def test_naive_sharing_lets_sybils_frame_honest_agents():
    df = Simulation(from_dict(_shared("naive"))).run()
    assert df["shared_global_blocks_honest"].iloc[-1] > 0


def test_alerts_never_block_globally():
    df = Simulation(from_dict(_shared("alerts"))).run()
    last = df.iloc[-1]
    assert last["shared_global_blocks_honest"] == 0 and last["shared_global_blocks_sybil"] == 0


def test_local_mode_matches_proposed_defense():
    cfg = _shared("local", fr=0)
    a = Simulation(from_dict(cfg)).run()
    cfg["defense"]["kind"] = "proposed"
    b = Simulation(from_dict(cfg)).run()
    assert (a["wrong_used_rate"] - b["wrong_used_rate"]).abs().max() < 1e-12


def test_real_card_mode_runs_when_bundle_exists():
    bundle = HERE.parents[3] / "data" / "real_cards" / "bundle.npz"
    if not bundle.exists():
        pytest.skip("real-card bundle not built")
    cfg = {"rounds": 3, "network": {"n_honest": 30, "capability_dim": 384, "real_cards": str(bundle)},
           "adversary": {"n_sybils": 10, "adaptive_cards": True}, "selection": {"kind": "reputation"},
           "defense": {"kind": "proposed", "params": {"n_anchors": 2}}}
    df = Simulation(from_dict(cfg)).run()
    assert len(df) == 3 and df["target_sybil_share"].between(0, 1).all()


def test_transcripts_clear_honest_agents_with_correct_answers():
    from sybagent.agent import Agent
    cfg = _shared("committee", b=5, fr=0)
    cfg["defense"]["params"]["evidence_mode"] = "transcripts"
    sim = Simulation(from_dict(cfg))
    sim.run()
    d, ctx = sim.defense, sim.ctx
    honest = ctx.honest_ids[0]
    member = ctx.agents[ctx.honest_ids[1]]
    tasks = [ctx.agents[honest].tasks[0] if ctx.agents[honest].tasks is not None else
             np.ones(cfg["network"].get("capability_dim", 16)) / 4.0 for _ in range(3)]
    items = [(t, Agent.correct_answer(t)) for t in tasks]
    for a in list(member.pinned):                      # make the member's anchors honest and accurate
        member.unpin(a)
    for a in ctx.honest_ids[2:5]:
        member.pin(a)
        ctx.agents[a].answer_correct_prob = 1.0
    assert not d._check_items(member, items, ctx)
    wrong = [(t, (Agent.correct_answer(t) + 1) % 8) for t in tasks]
    assert d._check_items(member, wrong, ctx)


def test_false_reports_with_transcripts_do_not_frame():
    cfg = _shared("committee", b=8, fr=2)
    cfg["rounds"] = 12
    cfg["defense"]["params"].update(evidence_mode="transcripts", committee_size=9)
    df = Simulation(from_dict(cfg)).run()
    assert df["shared_global_blocks_honest"].iloc[-1] <= 1


def test_proof_ban_bans_liars_not_honest():
    cfg = _shared("local", b=10, fr=0)
    cfg["rounds"] = 30
    cfg["defense"]["params"].update(audit_prob=0.0, verify_prob=0.5, proof_ban=True, verified_rep=True)
    df = Simulation(from_dict(cfg)).run()
    last = df.iloc[-1]
    assert last["shared_global_blocks_sybil"] >= 8 and last["shared_global_blocks_honest"] == 0
    assert last["verify_calls"] > 0


def test_sleeper_policy_switches():
    cfg = _small("sleeper", sleeper_round=3)
    sim = Simulation(from_dict(cfg))
    df = sim.run()
    assert df["strategy_lie_frac"].iloc[2] == 0.0 and df["strategy_lie_frac"].iloc[-1] > 0.0
