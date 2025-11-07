# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from sympy import Segment3D

from tools.frozen import Frozen


class Point3(Frozen):
    r"""The class define a 2D point."""
    def __init__(self, x, y, z):
        r"""

        Parameters
        ----------
        x : int, float
            x-coordinate
        y : int, float
            y-coordinate
        z : int, float
            z-coordinate

        """
        assert isinstance(x, (float, int))
        assert isinstance(y, (float, int))
        assert isinstance(z, (float, int))
        self._x = x
        self._y = y
        self._z = z
        self._coo = (x, y, z)
        self._freeze()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z


def distance3(*objs):
    r"""Compute distance between some things."""
    if len(objs) == 2:
        obj0, obj1 = objs
        if obj0.__class__ is Point3 and obj1.__class__ is Point3:
            # compute the distance between two Points.
            x0, y0, z0 = obj0.x, obj0.y, obj0.z
            x1, y1, z1 = obj1.x, obj1.y, obj1.z
            return np.sqrt((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2)
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()


class StraightLine3(Frozen):
    r"""A infinite line through A to B."""
    def __init__(self, A, B):
        r"""

        Parameters
        ----------
        A : Point3
        B : Point3
        """
        assert A.__class__ is Point3, f"I need a Point3 instance."
        assert B.__class__ is Point3, f"I need a Point3 instance."
        assert distance3(A, B) > 1e-5, f"the two points are too close!"
        self._A = A
        self._B = B

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B


class StraightSegment3(Frozen):
    r"""A line segment from A to B. It is not infinite."""
    def __init__(self, A, B):
        r"""

        Parameters
        ----------
        A : Point3
        B : Point3
        """
        assert A.__class__ is Point3, f"I need a Point3 instance."
        assert B.__class__ is Point3, f"I need a Point3 instance."
        assert distance3(A, B) > 1e-4, f"the two points are too close!"
        self._A = A
        self._B = B

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B


def angle3(A, o, B):
    r"""compute the angle of AoB"""
    assert A.__class__ is Point3, f"I need a Point3 instance."
    assert o.__class__ is Point3, f"I need a Point3 instance."
    assert B.__class__ is Point3, f"I need a Point3 instance."
    vec_oA = np.array([A.x - o.x, A.y - o.y, A.z - o.z])
    vec_oB = np.array([B.x - o.x, B.y - o.y, B.z - o.z])
    norm_oA = np.linalg.norm(vec_oA)
    norm_oB = np.linalg.norm(vec_oB)
    return np.arccos(vec_oA.dot(vec_oB) / (norm_oA * norm_oB))


class Polyhedron3(Frozen):
    r""""""
    def __init__(self, *vertices):
        r"""These vertices make up the polygon in their sequence."""
        for v in vertices:
            assert v.__class__ is Point3, f"vertices must be 3d points."
        self._vertices = vertices
        self.faces = {}
        self._freeze()


def whether_point_on_straight_segment(point, segment):
    r""""""
    assert point.__class__ is Point3, f"The point must be a Point3 instance in m3n3."
    assert segment.__class__ is StraightSegment3, f"segment must be straight in m3n3."
    A, B = segment.A, segment.B
    d = distance3(A, point) + distance3(B, point)
    dAB = distance3(A, B)
    if abs(d - dAB) < 1e-8:
        return True
    else:
        return False


if __name__ == '__main__':
    A = Point3(1, 0, 0)
    B = Point3(0, 1, 0)
    o = Point3(0, 0, 0)
    C = Point3(-1., 0.1, 0)

    print(angle3(B, o, A))

    s = StraightSegment3(A, C)
    print(whether_point_on_straight_segment(o, s))
