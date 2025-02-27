# -*- coding: utf-8 -*-
r"""
"""
import sys

if './' not in sys.path:
    sys.path.append('./')
from tools.frozen import Frozen


class ParallelLinesError(Exception):
    """Exception raised for custom error scenarios."""


class Point2(Frozen):
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        r"""X coordinate."""
        return self._x

    @property
    def y(self):
        """Y coordinate."""
        return self._y

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<Point ({self.x}, {self.y})" + super_repr


class Line2(Frozen):
    r""""""
    def __init__(self, A, B):
        r""""""
        assert A.__class__ is Point2
        assert B.__class__ is Point2
        self._A = A
        self._B = B

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B


def line2_line2_intersection(line1, line2):
    # Line AB represented as a1x + b1y = c1
    A, B = line1.A, line1.B
    C, D = line2.A, line2.B

    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1 * (A.x) + b1 * (A.y)

    # Line CD represented as a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2 * (C.x) + b2 * (C.y)

    determinant = a1 * b2 - a2 * b1

    if (determinant == 0):
        # The lines are parallel
        raise ParallelLinesError(f"Line AB is parallel to line CD.")
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return Point2(x, y)
