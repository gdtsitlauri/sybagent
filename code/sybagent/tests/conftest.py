import numpy as np
import pytest

from sybagent.config import SimConfig


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_cfg():
    cfg = SimConfig(seed=0, rounds=10)
    cfg.network.n_honest = 30
    cfg.network.max_neighbors = 4
    cfg.network.candidates_per_query = 8
    cfg.network.topk_select = 2
    cfg.adversary.n_sybils = 10
    return cfg
