import numpy
from rules import rule_180_with_signals

def test_rule_180_with_signals():
    matrix = numpy.array([1, 1, 0, 0, -1, 0]).reshape((1, -1))
    assert rule_180_with_signals(matrix, (1, 0, 1), [(-1, 0, 4)]) == (1, 0, 2)
    assert rule_180_with_signals(matrix, (1, 0, 0), [(1, 0, 1)]) == (1, 0, 0)
    assert rule_180_with_signals(matrix, (-1, 0, 4), []) == (-1, 0, 4)
