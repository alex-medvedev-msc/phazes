import numpy
from prototype import Changer


def test_changer():
    changer = Changer(0.5, 1, 1)
    matrix = numpy.array([[0, 0, 0]])
    changer.change(matrix)
    assert matrix[0, 2] == 0
    changer.change(matrix)
    assert matrix[0, 2] == -1
    changer.change(matrix)
    assert matrix[0, 2] == 0
