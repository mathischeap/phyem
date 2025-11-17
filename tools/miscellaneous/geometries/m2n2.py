# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from matplotlib.path import Path
from pynverse import inversefunc

from phyem.tools.frozen import Frozen
from phyem.tools.functions.space._2d.angle import angle
from phyem.tools.miscellaneous.geometries.m1n1 import Interval
from phyem.src.config import RANK, MASTER_RANK


class Parallel2dLinesError(Exception):
    """Exception raised for custom error scenarios."""


class Point2(Frozen):
    r"""The class define a 2D point."""
    def __init__(self, x, y):
        r"""

        Parameters
        ----------
        x : int, float
            x-coordinate
        y : int, float
            y-coordinate

        """
        self._x = x
        self._y = y
        self._coo = (x, y)
        self._freeze()

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

    @property
    def coo(self):
        return self._coo


class StraightLine2(Frozen):
    r"""Straight line."""
    def __init__(self, A, B):
        r"""

        Parameters
        ----------
        A : Point2
        B : Point2
        """
        assert A.__class__ is Point2, f"I need a Point2 instance."
        assert B.__class__ is Point2, f"I need a Point2 instance."
        assert distance2(A, B) > 1e-4, f"the two points are too close!"
        self._A = A
        self._B = B
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B


class StraightSegment2(Frozen):
    r"""Like a `StraightLine2` but it is not infinite."""
    def __init__(self, A, B):
        r"""

        Parameters
        ----------
        A :
        B :
        """
        if isinstance(A, (list, tuple)):
            A = Point2(*A)
        if isinstance(B, (list, tuple)):
            B = Point2(*B)

        assert A.__class__ is Point2, f"I need a Point2 instance."
        assert B.__class__ is Point2, f"I need a Point2 instance."
        assert distance2(A, B) > 1e-4, f"the two points are too close!"
        self._A = A
        self._B = B

        x_diff = abs(A.x - B.x)
        y_diff = abs(A.y - B.y)
        if x_diff > y_diff:
            check_coo = 'x'
            if A.x <= B.x:
                lb, ub = A.x, B.x
            else:
                lb, ub = B.x, A.x
        else:
            check_coo = 'y'
            if A.y <= B.y:
                lb, ub = A.y, B.y
            else:
                lb, ub = B.y, A.y
        self.___check_cache___ = (check_coo, lb, ub)

        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split(' at ')[1]
        return rf"{self.__class__.__name__}=({self.A},{self.B}) at " + super_repr


def distance2(*objs):
    r"""Compute distance between some things."""
    if len(objs) == 2:
        obj0, obj1 = objs
        if obj0.__class__ is Point2 and obj1.__class__ is Point2:
            # compute the distance between two Points.
            x0, y0 = obj0.x, obj0.y
            x1, y1 = obj1.x, obj1.y
            return np.sqrt((x1 - x0)**2 + (y1 - y0)**2)
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()


def whether_point_on_straight_line(point, line):
    r""""""
    assert point.__class__ is Point2, f"The point must be a Point2 instance in m2n2."
    assert line.__class__ is StraightLine2, f"line must be straight in m2n2."
    A, B = line.A, line.B
    if distance2(A, point) < 1e-8:
        return True
    elif distance2(B, point) < 1e-8:
        return True
    else:
        angle0 = angle(point.coo, A.coo)
        angle1 = angle(B.coo, point.coo)
        if np.isclose(angle0, angle1):
            return True
        elif (np.isclose(angle0 + np.pi, angle1)
              or np.isclose(angle0 - np.pi, angle1)
              or np.isclose(angle0, angle1 + np.pi)
              or np.isclose(angle0, angle1 - np.pi)):
            return True
        else:
            return False


def whether_point_on_straight_segment(point, segment):
    r""""""
    if isinstance(point, tuple):
        point = Point2(*point)
    else:
        pass
    assert point.__class__ is Point2, f"The point must be a Point2 instance in m2n2."
    assert segment.__class__ is StraightSegment2, f"segment must be straight in m2n2."
    A, B = segment.A, segment.B
    if distance2(A, point) < 1e-8:
        return True
    elif distance2(B, point) < 1e-8:
        return True
    else:
        angle0 = angle(point.coo, A.coo)
        angle1 = angle(B.coo, point.coo)
        if np.isclose(angle0, angle1):
            return True
        else:
            return False


def line2_line2_intersection(line_1, line_2):
    r"""Find the intersection of two straight lines, if it exists. Otherwise, raise ParallelLinesError.

    Parameters
    ----------
    line_1 : StraightLine2
    line_2 : StraightLine2

    Returns
    -------

    """
    # Line AB represented as a1x + b1y = c1
    A, B = line_1.A, line_1.B
    C, D = line_2.A, line_2.B

    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1 * A.x + b1 * A.y

    # Line CD represented as a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2 * C.x + b2 * C.y

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # The lines are parallel
        raise Parallel2dLinesError(f"Line AB is parallel to line CD.")
    else:
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return Point2(x, y)


class Polygon2(Frozen):
    r""""""
    def __init__(self, *vertices):
        r"""These vertices make up the polygon in their sequence."""
        for v in vertices:
            assert v.__class__ is Point2, f"vertices must be 2d points."
        self._vertices = vertices
        self._edges = {}
        self._path = None
        self._freeze()

    def __getitem__(self, i):
        r"""return the ith vertex."""
        return self._vertices[i]

    def __len__(self):
        r"""How many vertices this polygon has?"""
        return len(self._vertices)

    def __iter__(self):
        r""""""
        for v in self._vertices:
            yield v

    def edges(self, i):
        r"""Return the ith edge. The 0-th edge is from vertex0 to vertex 1. The 1-st edge is from vertex1 to vertex2.
        And the last edge is from the last vertex to vertex0.
        """
        if i in self._edges:
            pass
        else:
            assert i in range(len(self)), f"I only have {len(self)} edges."
            num_vertices = len(self)
            if i < num_vertices - 1:
                v0, v1 = self[i], self[i + 1]
            else:
                v0, v1 = self[num_vertices - 1], self[0]
            self._edges[i] = StraightLine2(v0, v1)
        return self._edges[i]

    @property
    def path(self):
        if self._path is None:
            # noinspection PyTypeChecker
            self._path = Path([(_.x, _.y) for _ in self])
        return self._path

    def visualize(self):
        r""""""
        if RANK == MASTER_RANK:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_aspect('equal')
            x = list()
            y = list()
            for v in self._vertices:
                x.append(v._x)
                y.append(v._y)
            x.append(self[0]._x)
            y.append(self[0]._y)
            plt.fill(x, y, c='lightgray')
            plt.plot(x, y, linewidth=1, c='k')
            plt.show()
            plt.close(fig)
        else:
            pass


def whether_point_in_polygon(point, polygon):
    r"""check whether a point is in a polygon or is on its edge.

    Parameters
    ----------
    point
    polygon

    Returns
    -------

    """
    num_vertices = len(polygon)

    on_edge = False
    for i in range(num_vertices):
        edge = polygon.edges(i)
        if whether_point_on_straight_line(point, edge):
            on_edge = True
            break
        else:
            pass

    if on_edge:
        return True
    else:
        return polygon.path.contains_point((point.x, point.y))


class Curve2(Frozen):
    r"""m2n2 curve."""

    def __init__(self, xt, yt, t_interval):
        r"""

        Parameters
        ----------
        xt :
            x(t) and it must be a 'monotone increasing' or 'monotone decreasing' function of t. We can compute t
            from x.
        yt :
            y(t) can be any kind of function of t; we cannot compute t from y.
        t_interval :
            The interval of t. It determines the interval of x.
        """
        if isinstance(t_interval, Interval):
            pass
        else:
            t_interval = Interval(t_interval)
        self._interval = t_interval
        x = xt(self._interval.linspace(33))
        xd = np.diff(x)
        assert all(xd >= 0) or all(xd <= 0), f"x(t) must be 'monotone increasing' or 'monotone decreasing'"
        x_min, x_max = min(x), max(x)
        if self._interval._type == 'open':
            x_interval = Interval((x_min, x_max))
        elif self._interval._type == 'closed':
            x_interval = Interval([x_min, x_max])
        else:
            raise Exception()
        self._x_interval = x_interval
        self._xt = xt
        self._yt = yt
        self._tx = inversefunc(xt)  # inverse function of xt, it must exist since x(t) is monotone!
        self._freeze()

    def x_from_t(self, t):
        assert t in self._interval, f"t or some elements of t is out of curve t range."
        return self._xt(t)

    def y_from_t(self, t):
        assert t in self._interval, f"t or some elements of t is out of curve t range."
        return self._yt(t)

    def t_from_x(self, x):
        r""""""
        assert x in self._x_interval, \
            f"x={x} (or some elements of x) is in out of curve x range: {self._x_interval}."
        return self._tx(x)

    @property
    def interval(self):
        r"""the range of t."""
        return self._interval

    def visualize(self):
        r""""""
        if RANK == MASTER_RANK:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_aspect('equal')
            interval = self._interval.linspace(33)
            x = self.x_from_t(interval)
            y = self.y_from_t(interval)
            plt.plot(x, y, linewidth=0.75, c='k')
            plt.show()
            plt.close(fig)
        else:
            pass


def whether_point_on_curve(point, curve):
    r""""""
    if isinstance(point, tuple):
        point = Point2(*point)
    else:
        pass
    assert point.__class__ is Point2, f"point must be a {Point2} instance."
    assert curve.__class__ is Curve2, f"curve must be a {Curve2} instance."

    x, y = point.x, point.y

    if x in curve._x_interval:
        t = curve.t_from_x(x)
        Y = curve.y_from_t(t)
        if np.isclose(y, Y):
            return True
        else:
            return False
    else:
        return False


if __name__ == "__main__":
    # Define a point to test
    point = Point2(1.01, 0.5)

    # Define a polygon
    polygon = Polygon2(
        Point2(0, 0),
        Point2(1, 0),
        Point2(1, 1),
        Point2(0, 1)
    )

    # print(whether_point_in_polygon(point, polygon))

    # def f(x):
    #     return x ** 3
    #
    # inverse = inversefunc(f)
    # print(inverse(-27))


    def x(t):
        return t


    def y(t):
        return np.sin(t)


    t_range = [1, 2]

    curve = Curve2(x, y, t_range)
    curve.visualize()
