from sybagent.metrics import honest_neighbor_count, sybil_share, time_to_eclipse


def test_sybil_share():
    assert sybil_share([], {1}) == 0.0
    assert sybil_share([1, 2, 3, 4], {1, 2}) == 0.5


def test_honest_neighbor_count():
    assert honest_neighbor_count([1, 2, 3], {2}) == 2


def test_time_to_eclipse():
    rows = [{"round": 0, "targets_eclipsed_frac": 0.0}, {"round": 1, "targets_eclipsed_frac": 1.0}]
    assert time_to_eclipse(rows) == 1
    assert time_to_eclipse(rows[:1]) is None
