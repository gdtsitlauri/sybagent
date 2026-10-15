import pytest

from sybagent.selection.llm import LLMSelector
from sybagent.simulation import Simulation

BASELINES = ["none", "degree_bound", "random_mix", "sybilrank", "sumup", "proposed"]


@pytest.mark.parametrize("defense", BASELINES)
def test_runs_with_each_defense(small_cfg, defense):
    small_cfg.defense.kind = defense
    df = Simulation(small_cfg).run()
    assert len(df) == small_cfg.rounds
    assert df["target_sybil_share"].between(0, 1).all()
    assert (df["n_sybils"] == small_cfg.adversary.n_sybils).all()


def test_no_adversary_means_no_exposure(small_cfg):
    small_cfg.adversary.enabled = False
    df = Simulation(small_cfg).run()
    assert (df["honest_mean_sybil_share"] == 0).all()


def test_deterministic_given_seed(small_cfg):
    a = Simulation(small_cfg).run()
    b = Simulation(small_cfg).run()
    assert a.equals(b)


def test_proposed_keeps_anchor_slots(small_cfg):
    small_cfg.defense.kind = "proposed"
    small_cfg.defense.params = {"n_anchors": 2}
    sim = Simulation(small_cfg)
    sim.run()
    for aid in sim.ctx.honest_ids:
        a = sim.ctx.agents[aid]
        assert len(a.pinned) == 2
        assert len(a.peers) <= small_cfg.network.max_neighbors - 2
        assert len(a.neighbors) <= small_cfg.network.max_neighbors


def test_sleeper_sybil_anchors_get_evicted(small_cfg):
    small_cfg.rounds = 60
    small_cfg.defense.kind = "proposed"
    small_cfg.defense.params = {"n_anchors": 2, "anchor_use": 1.0}
    small_cfg.adversary.n_sybils = 60
    small_cfg.adversary.service_quality = 0.85
    small_cfg.adversary.switch_round = 20
    small_cfg.adversary.post_switch_quality = 0.1
    sim = Simulation(small_cfg)

    def sybil_anchors():
        return sum(p in sim.ctx.sybil_ids for a in sim.ctx.honest_ids for p in sim.ctx.agents[a].pinned)

    for _ in range(20):
        sim.step()
    before = sybil_anchors()
    for _ in range(40):
        sim.step()
    assert before > 0
    assert sybil_anchors() < before / 2


def test_audit_blocks_wrong_answering_sybils(small_cfg):
    small_cfg.rounds = 40
    small_cfg.defense.kind = "proposed"
    small_cfg.defense.params = {"n_anchors": 3, "audit_prob": 1.0, "audit_flag": 3}
    small_cfg.adversary.n_sybils = 10
    small_cfg.adversary.service_quality = 0.85
    small_cfg.adversary.answer_wrong_prob = 0.5
    small_cfg.adversary.adaptive_cards = True
    sim = Simulation(small_cfg)
    df = sim.run()
    blocked = sim.defense._blocked
    sy = sim.ctx.sybil_ids
    n_sybil_blocks = sum(len(b & sy) for b in blocked.values())
    n_honest_blocks = sum(len(b - sy) for b in blocked.values())
    assert n_sybil_blocks > 5 * max(1, n_honest_blocks)
    assert df["audit_calls"].iloc[-1] > 0


def test_audit_off_keeps_selection_stream(small_cfg):
    small_cfg.defense.kind = "proposed"
    a = Simulation(small_cfg).run()
    small_cfg.adversary.answer_wrong_prob = 0.9      # only affects the separate answer stream
    b = Simulation(small_cfg).run()
    answer_cols = ("wrong_answer_rate", "wrong_used_rate", "sybil_wrong_served", "bad_corrections", "good_corrections")
    cols = [c for c in a.columns if c not in answer_cols]
    assert a[cols].equals(b[cols])


def test_unanimous_rule_resists_coordinated_framing(small_cfg):
    """Coordinated lying Sybil anchors can flip correct answers under majority, much less under unanimity."""
    def run(rule):
        cfg = small_cfg
        cfg.rounds = 40
        cfg.network.answer_model = "skill"
        cfg.defense.kind = "proposed"
        cfg.defense.params = {"n_anchors": 3, "audit_prob": 1.0, "audit_flag": 3, "audit_rule": rule}
        cfg.adversary.n_sybils = 30                   # Sybils outnumber the 30 honest agents' share of anchors
        cfg.adversary.answer_wrong_prob = 1.0
        cfg.adversary.coordinated_lies = True
        cfg.adversary.service_quality = 0.85
        df = Simulation(cfg).run()
        return int(df["bad_corrections"].sum()), int(df["blocked_honest_pairs"].iloc[-1])
    bad_maj, honest_maj = run("majority")
    bad_una, honest_una = run("unanimous")
    assert bad_una < bad_maj
    assert honest_una <= honest_maj


def test_category_anchors_come_from_category(small_cfg):
    small_cfg.defense.kind = "proposed"
    small_cfg.defense.params = {"n_anchors": 3, "anchor_category_frac": 1.0}
    small_cfg.adversary.enabled = False
    sim = Simulation(small_cfg)
    sim.run()
    import numpy as np
    kinds = sim.defense._kind
    n_cat = 0
    for aid in sim.ctx.honest_ids:
        a = sim.ctx.agents[aid]
        cat = int(np.argmax(a.profile))
        for p in a.pinned:
            if kinds[(aid, p)] == "cat":      # small categories fall back to global anchors
                n_cat += 1
                assert int(np.argmax(sim.ctx.agents[p].card)) == cat
    assert n_cat > 0


def test_anchor_bound():
    from sybagent.defenses.proposed import anchor_eclipse_bound
    assert anchor_eclipse_bound(100, 0, 3) == 0.0
    assert abs(anchor_eclipse_bound(102, 100, 2) - 0.25) < 1e-12
    assert anchor_eclipse_bound(100, 100, 2, beta=1.0) == 1.0


def test_llm_id_parsing():
    assert LLMSelector.parse_ids('{"ids": [4, 9, 4]}', allowed=[4, 9, 12], k=2) == [4, 9]
    assert LLMSelector.parse_ids("I pick agent 12 and agent 99", allowed=[4, 9, 12], k=2) == [12]
