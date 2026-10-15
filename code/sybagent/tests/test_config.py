from pathlib import Path

import pytest

from sybagent.config import from_dict, load_config, merge
from sybagent.experiments.runner import expand_sweep


def test_from_dict_nested():
    cfg = from_dict({"rounds": 5, "network": {"n_honest": 7}, "defense": {"kind": "sybilrank", "params": {"n_seeds": 3}}})
    assert cfg.rounds == 5
    assert cfg.network.n_honest == 7
    assert cfg.network.max_neighbors == 8
    assert cfg.defense.params == {"n_seeds": 3}


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        from_dict({"network": {"bogus": 1}})


def test_merge_is_recursive():
    assert merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_expand_sweep_counts():
    spec = {"base": {"rounds": 2}, "grid": {"adversary.n_sybils": [0, 5], "defense.kind": ["none", "random_mix"]}, "seeds": [0, 1, 2]}
    runs = expand_sweep(spec)
    assert len(runs) == 12
    assert {r["adversary"]["n_sybils"] for r in runs} == {0, 5}


def test_default_yaml_loads():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "default.yaml")
    assert cfg.network.n_honest == 200
