import numpy as np
import pytest
from helpers import random_small

from pgt import Instance, Schedule, left_justify


@pytest.fixture
def inst() -> Instance:
    return Instance(p=3, r=[0, 1, 7], d=[3, 4, 9], w=[1, 2, 5])


def test_objective(inst):
    s = Schedule([0, 3, 7])
    assert s.is_feasible(inst)
    assert s.tardiness(inst).tolist() == [0, 2, 1]
    assert s.F(inst) == 2 * 2 + 5 * 1
    assert s.Phi(inst) == s.F(inst) + 3 * 8


def test_violations(inst):
    assert Schedule([0, 0, 7]).violations(inst)  # job 1 before release, overlap
    assert not Schedule([0, 3, 6]).is_feasible(inst)  # job 2 before release
    assert not Schedule([0, 2, 7]).is_feasible(inst)  # overlap
    assert Schedule([4, 1, 7]).is_feasible(inst)  # adjacent intervals touch, fine
    assert not Schedule([0, 3]).is_feasible(inst)


def test_left_justify(inst):
    assert left_justify(inst, [0, 1, 2]).starts.tolist() == [0, 3, 7]
    assert left_justify(inst, [2, 1, 0]).starts.tolist() == [13, 10, 7]
    assert left_justify(inst, [1, 0, 2]).starts.tolist() == [4, 1, 7]
    with pytest.raises(ValueError):
        left_justify(inst, [0, 0, 1])


def test_left_justify_never_worse(rng):
    for _ in range(200):
        inst = random_small(rng)
        # random feasible schedule: random order with random idle gaps
        starts, free = np.empty(inst.n, dtype=np.int64), 0
        for j in rng.permutation(inst.n):
            starts[j] = max(int(inst.r[j]), free) + int(rng.integers(0, 2 * inst.p + 1))
            free = starts[j] + inst.p
        sched = Schedule(starts)
        assert sched.is_feasible(inst)
        lj = left_justify(inst, sched.order())
        assert lj.is_feasible(inst)
        assert (lj.starts <= sched.starts).all()
        assert lj.F(inst) <= sched.F(inst)


def test_json_roundtrip(tmp_path, inst):
    inst.save(tmp_path / "i.json")
    assert Instance.load(tmp_path / "i.json") == inst
    s = Schedule([0, 3, 7])
    s.save(tmp_path / "s.json")
    assert Schedule.load(tmp_path / "s.json") == s


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(p=0, r=[0], d=[0], w=[1]),
        dict(p=1, r=[0, 1], d=[0], w=[1]),
        dict(p=1, r=[], d=[], w=[]),
        dict(p=1, r=[-1], d=[0], w=[1]),
        dict(p=1, r=[0], d=[0], w=[0]),
        dict(p=1, r=[0.5], d=[0], w=[1]),
        dict(p=1.0, r=[0], d=[0], w=[1]),
    ],
)
def test_instance_validation(kwargs):
    with pytest.raises(ValueError):
        Instance(**kwargs)


def test_instance_rejects_int64_overflow():
    with pytest.raises(OverflowError):
        Instance(p=1, r=[0], d=[0], w=[2**63])
