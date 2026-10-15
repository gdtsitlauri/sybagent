import numpy as np

from sybagent.agent import Agent, unit
from sybagent.registry import Registry, ReviewLedger


def _agent(aid, vec, honest=True, k=3):
    v = unit(np.asarray(vec, dtype=float))
    return Agent(aid=aid, honest=honest, capability=v, card=v, max_neighbors=k)


def test_working_set_evicts_oldest():
    a = _agent(0, [1, 0])
    for p in [1, 2, 3, 4]:
        a.remember(p, 0.9)
    assert a.neighbors == [2, 3, 4]


def test_remember_running_mean_and_prune():
    a = _agent(0, [1, 0])
    a.remember(1, 0.2)
    a.remember(1, 0.2)
    a.remember(2, 0.9)
    dropped = a.prune(below=0.3, min_obs=2)
    assert dropped == [1]
    assert a.neighbors == [2]


def test_registry_query_ranks_by_similarity():
    reg = Registry()
    reg.register(_agent(0, [1, 0]))
    reg.register(_agent(1, [0, 1]))
    reg.register(_agent(2, [1, 1]))
    assert reg.query(unit(np.array([1.0, 0.0])), 2) == [0, 2]
    assert reg.query(unit(np.array([1.0, 0.0])), 2, exclude=0) == [2, 1]


def test_ledger_mean_with_prior():
    led = ReviewLedger()
    assert led.mean(5) == 0.5
    led.add(1, 5, 1.0)
    assert led.mean(5) == 0.75
