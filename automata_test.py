import numpy
from automata import rule_180, apply_rule, find_agent, full_neighborhood, iterate, simulate
import pytest


def test_rule180():
    matrix = numpy.array([1, 0, 1, 0, 1, 1]).reshape((1, -1))
    assert rule_180(matrix, (1, 0, 2), []) == (1, 0, 3)
    assert rule_180(matrix, (1, 0, 5), [(1, 0, 6)]) == (1, 0, 5)
    assert rule_180(matrix, (1, 0, 6), []) == (1, 0, 7)


def test_apply_rule():
    matrix = numpy.array([1, 0, 1, 0, 0, 1, 1]).reshape((1, -1))
    desired = numpy.array([0, 1, 0, 1, 0, 1, 0]).reshape((1, -1))
    new_matrix = numpy.zeros(matrix.shape, dtype=matrix.dtype)

    assert not apply_rule(matrix, new_matrix, (1, 0, 0), [], rule_180)
    assert not apply_rule(matrix, new_matrix, (1, 0, 2), [], rule_180)
    assert not apply_rule(matrix, new_matrix, (1, 0, 5), [(1, 0, 6)], rule_180)
    assert not apply_rule(matrix, new_matrix, (1, 0, 6), [], rule_180)
    assert numpy.equal(new_matrix, desired).all()

    with pytest.raises(ValueError):
        apply_rule(matrix, new_matrix, (1, 0, 0), [], rule_180)
    with pytest.raises(ValueError):
        apply_rule(matrix, new_matrix, (1, 1, 0), [], rule_180)


def test_find_agent():
    lanes = [numpy.array([0, 0, 0, 1]), numpy.array([1]), numpy.array([0, 1, 0, 0]), numpy.array([0, 0, 0, 0])]
    desired = [3, 0, 1, None]
    desired_reversed = [0, 0, 2, None]
    for lane, index in zip(lanes, desired):
        agent = find_agent(lane, False)
        if agent is None:
            assert agent == index
            continue
        state, i = agent
        assert i == index
    for lane, index in zip(lanes, desired_reversed):
        agent = find_agent(lane, True)
        if agent is None:
            assert agent == index
            continue
        state, i = agent
        assert i == index


def test_full_neighborhood():
    matrix = numpy.array([[1, 0, 2, 0, 0, 0, 3], [4, 5, 0, 0, 6, 0, 0], [0, 0, 0, 7, 0, 0, 0]])
    agents = [(1, 0, 0), (2, 0, 2), (3, 0, 6), (4, 1, 0), (5, 1, 1), (6, 1, 4), (7, 2, 3)]
    desired = [
        [2, 4],
        [1, 3, 5, 6],
        [2, 6, 7],
        [5, 1, 7],
        [1, 2, 4, 6, 7],
        [2, 3, 5, 7],
        [5, 6, 2]
    ]
    for agent, desired_nb in zip(agents, desired):
        nb = full_neighborhood(matrix, agent)
        a1 = sorted([a[0] for a in nb])
        a2 = sorted(desired_nb)
        assert len(a1) == len(a2)
        assert numpy.equal(a1, a2).all()


def test_iterate():
    matrix = numpy.array([[1, 0, 2, 0, 0, 0, 3], [4, 5, 0, 0, 6, 0, 0], [0, 0, 0, 7, 0, 0, 0]])
    desired = numpy.array([[0, 1, 0, 2, 0, 0, 0], [4, 0, 5, 0, 0, 6, 0], [0, 0, 0, 0, 7, 0, 0]])
    result = iterate(matrix, rule_180)
    assert numpy.equal(result, desired).all()


def test_simulate():
    matrix = numpy.array([[1, 0, 2, 0, 0, 0, 3], [4, 5, 0, 0, 6, 0, 0], [0, 0, 0, 7, 0, 0, 0]])
    desired = numpy.array([[0, 0, 1, 0, 2, 0, 0], [0, 4, 0, 5, 0, 0, 6], [0, 0, 0, 0, 0, 7, 0]])
    result = simulate(start=matrix, steps=2, rule=rule_180)
    assert numpy.equal(result, desired).all()