# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
import math
from matplotlib.path import Path
from pynverse import inversefunc
import geopandas as gpd
from shapely.geometry import Polygon as shapelyPolygon
from shapely.ops import unary_union

from phyem.tools.frozen import Frozen
# from phyem.tools.functions.space._2d.angle import angle
from phyem.tools.miscellaneous.geometries.m1n1 import Interval
from phyem.src.config import RANK, MASTER_RANK


class Parallel2dLinesError(Exception):
    """Exception raised for custom error scenarios."""


class NotParallel2dLinesError(Exception):
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
    def signature(self):
        r"""Same objects have a same signature."""
        return "Point2@(%.8f" % round(self.x, 8) + ',' + "%.8f)" % round(self.y, 8)

    @property
    def coo(self):
        return self._coo

    def ___possible_subset_group_key___(self):
        r"""Return a key-string, and other objects of the same key-string can be a subset
        of me. Or the other way around.
        """
        return "p%.6f" % round(self._x, 8)

    def __eq__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            if (math.isclose(self.x, other.x, abs_tol=1e-9) and
                    math.isclose(self.y, other.y, abs_tol=1e-9)):
                return True
            else:
                return False
        else:
            raise NotImplementedError(
                f"not implemented for equality between "
                f"{self.__class__.__name__} and {other} ({other.__class__.__name__})")

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r""""""
        if geo.__class__ is not self.__class__:
            return False
        else:
            return self == geo

    def is_equal_to_a_union_of_a_part_of_separate_geometries_of_same_type(self, geometries):
        r"""check whether this geo is equal to the Union of some of `geometries`.

        Remember, it must be `equal`, not be a subset. So, we must could find some geometries from `geometries`
        whose union is exactly equal to self.

        `separate` means intersection of geometries can only be None or a geo of a lower dimension than mine.

        `geometries` must be a list or tuple of geometries of the same type as self.

        In this case, since I am a 2d point, we just need to check if this point presents in `geometries`.

        And since they are separate, so `geometries` must contain no points equal to each other.
        """
        for i, geo in enumerate(geometries):
            assert geo.__class__ is self.__class__, f"geometries[{i}]={geo} is not a {self.__class__}"
            for j, GEO in enumerate(geometries):
                if j > i:
                    if geo == GEO:
                        raise Exception(f"geometries[{i}] and geometries[{j}] is the same point!")
                    else:
                        pass
                else:
                    pass
        for geo in geometries:
            if geo == self:
                return True
            else:
                pass
        return False


class StraightLine2(Frozen):
    r"""Straight line (infinite)."""
    def __init__(self, A, B):
        r"""

        Parameters
        ----------
        A : Point2
        B : Point2
        """
        assert A.__class__ is Point2, f"I need a Point2 instance."
        assert B.__class__ is Point2, f"I need a Point2 instance."
        assert distance2(A, B) > 1e-3, f"the two points are too close!"
        self._A = A
        self._B = B
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def slope(self):
        r"""the slope of this line."""
        x0, y0 = self.A.x, self.A.y
        x1, y1 = self.B.x, self.B.y
        if x0 == x1:
            return np.inf
        else:
            return (y1 - y0) / (x1 - x0)


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
        self._length = distance2(A, B)
        assert self._length > 1e-3, f"the two points are too close!"
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

    @property
    def slope(self):
        r"""the slope of this line."""
        x0, y0 = self.A.x, self.A.y
        x1, y1 = self.B.x, self.B.y
        if abs(x0 - x1) < 1e-8:
            if y1 > y0:
                return - np.inf
            else:
                return np.inf
        else:
            slope = (y1 - y0) / (x1 - x0)
            return slope

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split(' at ')[1]
        return rf"{self.__class__.__name__}=({self.A},{self.B}) at " + super_repr

    def __eq__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            if self.A == other.A and self.B == other.B:
                return True
            elif self.A == other.B and self.B == other.A:
                # line segment has no starting point or ending point. They are the same.
                # AB = BA
                return True
            else:
                return False
        else:
            raise NotImplementedError(
                f"not implemented for equality between "
                f"{self.__class__.__name__} and {other} ({other.__class__.__name__})")

    def ___possible_subset_group_key___(self):
        r"""Return a key-string, and other objects of the same key-string can be a subset
        of me. Or the other way around.
        """
        slope = self.slope
        if abs(slope) == np.inf:
            return "ss:inf%.6f" % round(self.A.x, 8)
        else:
            if abs(slope) < 1e-8:
                slope = 0
            else:
                pass
            x0, y0 = self.A.x, self.A.y
            y = y0 - slope * x0
            return f"ss%.6f" % round(y, 8) + ":%.6f" % round(slope, 8)

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Return True iff `geo` is also a StraightSegment and it is a subset of self."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            if self == geo:
                return True
            else:
                if whether_point_on_straight_segment(geo.A, self):
                    return whether_point_on_straight_segment(geo.B, self)
                else:
                    return False

    def is_equal_to_a_union_of_a_part_of_separate_geometries_of_same_type(self, geometries):
        r"""check whether this geo is equal to the Union of some of `geometries`.

        `separate` means intersection of geometries can only be None or a geo of a lower dimension than mine.

        Remember, it must be `equal`, not be a subset. So, we must could find some geometries from `geometries`
        whose union is exactly equal to self.

        `geometries` must be a list or tuple of geometries of the same type as self.

        We will skip the non-parallel segments in geometries since they have no effect at all. So, even some
        of them are not separate, we do not care.
        """
        parallel_geometries = []
        for geo in geometries:
            if whether_two_lines_or_segments_parallel(self, geo):
                parallel_geometries.append(geo)
            else:
                pass

        # -------- check whether all geometries are separate ---------------------------
        for i, geo0 in enumerate(parallel_geometries):
            for j, geo1 in enumerate(parallel_geometries):
                if j > i:
                    intersection = find_intersection_of_two_parallel_straight_segments(geo0, geo1)
                    assert intersection is None or isinstance(intersection, Point2), \
                        f"some segments are not separate."
                else:
                    pass

        # ---------- method 1 ---------------------------------------------------------------
        possible_geometries = []
        for geo1 in parallel_geometries:
            if self.whether_contain__input__as_a_part_of_same_type(geo1):
                possible_geometries.append(geo1)
            else:
                pass

        # !!!!!!!!!! we assume that all geometries in possible_geometries are separate!!!
        # !!!!!!!!!! SO we do no more checks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if abs(self._length - sum([_._length for _ in possible_geometries])) < 1e-8:
            return True
        else:
            return False

        # ---------- method 2 ---------------------------------------------------------------
        # intersection_segments = []
        # for geo in parallel_geometries:
        #     intersection = find_intersection_of_two_parallel_straight_segments(self, geo)
        #     if intersection is None:
        #         pass
        #     elif isinstance(intersection, Point2):
        #         pass
        #     elif isinstance(intersection, self.__class__):
        #         intersection_segments.append(geo)
        #     else:
        #         raise Exception(f"the intersection of two parallel segments must be None, Point2, or a segment.")
        #
        # intersection_segments.append(self)  # all self to the group.
        # checking_dict = {}
        # # the idea is the A, B points of segments in intersection_segments will all appear twice only.
        # for geo in intersection_segments:
        #     As = geo.A.signature
        #     Bs = geo.B.signature
        #     if As in checking_dict:
        #         checking_dict[As] += 1
        #     else:
        #         checking_dict[As] = 1
        #     if Bs in checking_dict:
        #         checking_dict[Bs] += 1
        #     else:
        #         checking_dict[Bs] = 1
        #
        # for signature in checking_dict:
        #     if checking_dict[signature] == 2:
        #         pass
        #     else:
        #         return False
        # return True


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
    d1 = distance2(A, point)

    if d1 < 1e-8:
        return True
    else:
        d2 = distance2(B, point)
        if d2 < 1e-8:
            return True
        else:
            dAB = distance2(A, B)
            if math.isclose(d1 + d2, dAB, abs_tol=1e-8):
                return True
            elif math.isclose(abs(d1 - d2), dAB, abs_tol=1e-8):
                return True
            else:
                return False
            # angle0 = angle(point.coo, A.coo)
            # angle1 = angle(B.coo, point.coo)
            # if np.isclose(angle0, angle1):
            #     return True
            # elif (np.isclose(angle0 + np.pi, angle1)
            #       or np.isclose(angle0 - np.pi, angle1)
            #       or np.isclose(angle0, angle1 + np.pi)
            #       or np.isclose(angle0, angle1 - np.pi)):
            #     return True
            # else:
            #     return False


# ___2pi___ = 2 * np.pi


def whether_point_on_straight_segment(point, segment):
    r""""""
    if isinstance(point, tuple):
        point = Point2(*point)
    else:
        pass
    assert point.__class__ is Point2, f"The point must be a Point2 instance in m2n2."
    assert segment.__class__ is StraightSegment2, f"segment must be straight in m2n2."
    A, B = segment.A, segment.B
    d1 = distance2(A, point)
    if d1 < 1e-8:
        return True
    else:
        d2 = distance2(B, point)
        if d2 < 1e-8:
            return True
        else:
            dAB = distance2(A, B)
            if math.isclose(d1 + d2, dAB, abs_tol=1e-8):
                return True
            else:
                return False
            # angle0 = angle(point.coo, A.coo)
            # angle1 = angle(B.coo, point.coo)
            # if math.isclose(angle0, angle1, abs_tol=1e-7):
            #     return True
            # elif math.isclose(abs(angle0-angle1), ___2pi___, abs_tol=1e-7):
            #     return True
            # else:
            #     return False


def find_intersection_of_two_parallel_straight_segments(sg0, sg1):
    r""""""
    if whether_two_lines_or_segments_parallel(sg0, sg1):
        pass
    else:
        raise NotParallel2dLinesError(f"two segments are not parallel.")

    A0 = sg0.A
    B0 = sg0.B
    A1 = sg1.A
    B1 = sg1.B

    A0_on_sg1 = whether_point_on_straight_segment(A0, sg1)
    B0_on_sg1 = whether_point_on_straight_segment(B0, sg1)
    A1_on_sg0 = whether_point_on_straight_segment(A1, sg0)
    B1_on_sg0 = whether_point_on_straight_segment(B1, sg0)

    if A0_on_sg1 and B0_on_sg1:
        return sg0
    elif A1_on_sg0 and B1_on_sg0:
        return sg1
    elif (not A0_on_sg1) and (not B0_on_sg1) and (not A1_on_sg0) and (not B1_on_sg0):
        return None
    else:
        pass
    if A0_on_sg1:  # must not B0_on_sg1
        if A0 == A1 or A0 == B1:
            return A0
        else:
            if A1_on_sg0:
                return StraightSegment2(A0, A1)
            else:
                assert B1_on_sg0, f"Must be"
                return StraightSegment2(A0, B1)
    else:
        assert B0_on_sg1, f"must be!"
        if B0 == A1 or B0 == B1:
            return B0
        else:
            if A1_on_sg0:
                return StraightSegment2(B0, A1)
            else:
                assert B1_on_sg0, f"Must be"
                return StraightSegment2(B0, B1)


def whether_two_lines_or_segments_parallel(line_or_sg_0, line_or_sg_1):
    r"""Do what it says."""
    # Line AB represented as a1x + b1y = c1
    A, B = line_or_sg_0.A, line_or_sg_0.B
    C, D = line_or_sg_1.A, line_or_sg_1.B

    a1 = B.y - A.y
    b1 = A.x - B.x

    # Line CD represented as a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x

    determinant = a1 * b2 - a2 * b1

    if math.isclose(determinant, 0, abs_tol=1e-9):
        # The lines are parallel
        return True
    else:
        return False


def line2_line2_intersection(line_1, line_2):
    r"""Find the intersection of two straight lines, if it exists. Otherwise, raise ParallelLinesError.

    Do not use it for segment the intersection of two segments could be another segments. So when they
    are parallel, it is ok.

    Parameters
    ----------
    line_1 : StraightLine2
    line_2 : StraightLine2

    Returns
    -------

    """
    assert line_1.__class__ is StraightLine2 and line_2.__class__ is StraightLine2, \
        f"line1 and line2 must both be straight lines (not even straight segments)."

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

    if math.isclose(determinant, 0, abs_tol=1e-9):
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
        self._shapelyPolygon_ = None
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

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split(' at ')[1]
        v_coo = []
        for v in self._vertices:
            v_coo.append("(%.2f" % round(v.x, 2) + ",%.2f)" % round(v.y, 2))
        return rf"Polygon2:"+'-'.join(v_coo)+' at ' + super_repr

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
            self._edges[i] = StraightSegment2(v0, v1)
        return self._edges[i]

    @property
    def path(self):
        if self._path is None:
            # noinspection PyTypeChecker
            self._path = Path([(_.x, _.y) for _ in self])
        return self._path

    @property
    def area(self):
        r"""The area of this polygon."""
        return self.shapely_Polygon.area

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

    def _whether_contain_point_(self, point):
        r""""""
        return whether_point_in_polygon(point, self)

    def __eq__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            self__vertices = self._vertices
            other_vertices = other._vertices
            if len(self__vertices) == len(other_vertices):
                for ps, po in zip(self__vertices, other_vertices):
                    if ps == po:
                        pass
                    else:
                        return False
                return True
            else:
                return False
        else:
            raise NotImplementedError(
                f"not implemented for equality between "
                f"{self.__class__.__name__} and {other} ({other.__class__.__name__})")

    @property
    def shapely_Polygon(self):
        r""""""
        if self._shapelyPolygon_ is None:
            points = list()
            for v in self._vertices:
                points.append((v.x, v.y))
            self._shapelyPolygon_ = shapelyPolygon(points)
        return self._shapelyPolygon_

    def union(self, poly):
        r"""return a union of me and another polygon `poly`.

        If intersection of self and `poly` is empty, saying we will have a multi-polygon, then raise Error!
        """
        if poly.__class__ is self.__class__:
            polygon1 = self.shapely_Polygon
            polygon2 = poly.shapely_Polygon
        else:
            raise NotImplementedError()
        gdf = gpd.GeoDataFrame({'geometry': [polygon1, polygon2]})
        union_polygon = unary_union(gdf.geometry)
        if union_polygon.__class__.__name__ == 'MultiPolygon':
            return f'MultiPolygon-NotImplemented!'  # basically, a not implemented error
        else:
            # noinspection PyUnresolvedReferences
            points = np.array(union_polygon.exterior.coords)
            Points = list()
            for v in points:
                Points.append(Point2(*v))
            return self.__class__(*Points)

    def intersection(self, poly):
        r"""return the intersection of me and another polygon `poly`.

        If intersection of self and `poly` is empty, saying we will have a multi-polygon, then raise Error!
        """
        if poly.__class__ is self.__class__:
            polygon1 = self.shapely_Polygon
            polygon2 = poly.shapely_Polygon
        else:
            raise NotImplementedError()
        intersection = polygon1.intersection(polygon2)
        if intersection.__class__.__name__ == 'Point':
            # noinspection PyUnresolvedReferences
            return Point2(intersection.x, intersection.y)
        elif intersection.__class__.__name__ == 'LineString':
            x, y = intersection.coords.xy
            return f'LineString@x{x}y{y}'  # basically, a not implemented error
        else:
            # noinspection PyUnresolvedReferences
            points = np.array(intersection.exterior.coords)
            Points = list()
            for v in points:
                Points.append(Point2(*v))
            return self.__class__(*Points)

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Whether self contain another Polygon2 i.e. `geo`?
        If self is equal to `geo`, it is also called 'contained'."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            if self == geo:
                return True
            else:
                # intersection = self.intersection(geo)
                # if intersection.__class__ is self.__class__:
                #     area = intersection.shapely_Polygon.area
                #     if math.isclose(area, geo.shapely_Polygon.area, abs_tol=1e-8):
                #         return True
                #     else:
                #         return False
                # else:
                #     return False
                for v in geo._vertices:
                    if self._whether_contain_point_(v):
                        pass
                    else:
                        return False
                return True

    def is_equal_to_a_union_of_a_part_of_separate_geometries_of_same_type(self, geometries):
        r"""check whether this geo is equal to the Union of some of `geometries`.

        Remember, it must be `equal`, not be a subset. So, we must find some geometries from `geometries`
        whose union is exactly equal to self. Then return True, otherwise return False.

        `separate` means intersection of geometries can only be None or a geo of a lower dimension than mine.

        `geometries` must be a list or tuple of geometries of the same type as self.

        And since they are separate, so `geometries` must only contain polygons that the intersection any of two of
        them is None, Point2, or StraightSegment2.
        """
        for i, geo in enumerate(geometries):
            if geo.__class__ is not self.__class__:
                raise Exception(f"geometries[{i}]={geo} is not a polygon.")

        intersections = []
        for geo in geometries:
            its = self.intersection(geo)
            if its.__class__ is self.__class__:
                intersections.append(geo)
            else:
                pass

        AREA = 0
        for geo in intersections:
            AREA += geo.shapely_Polygon.area

        if math.isclose(AREA, self.shapely_Polygon.area, abs_tol=1e-8):
            pass
        else:
            return False

        for geo in intersections:
            for v in geo._vertices:
                if self._whether_contain_point_(v):
                    pass
                else:
                    return False

        for i, geo0 in enumerate(intersections):
            for j, geo1 in enumerate(intersections):
                if j > i:
                    sec = geo0.intersection(geo1)
                    if sec.__class__ is self.__class__:
                        if len(sec._vertices) == 0:
                            pass
                        elif math.isclose(sec.shapely_Polygon.area, 0, abs_tol=1e-8):
                            pass
                        else:
                            return False
                    else:
                        pass

        return True


def whether_point_in_polygon(point, polygon):
    r"""check whether a point is in a polygon or is on its edge.

    Parameters
    ----------
    point
    polygon

    Returns
    -------

    """
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) == 2:
        point = Point2(*point)
    else:
        assert isinstance(point, Point2), f"point={point} ({point.__class__.__name__}) is not a m2n2 point."

    num_vertices = len(polygon)

    for i in range(num_vertices):
        edge = polygon.edges(i)
        if whether_point_on_straight_segment(point, edge):
            return True
        else:
            pass

    ToF = polygon.path.contains_point((point.x, point.y))

    # if not ToF:
    #     print((point.x, point.y), polygon)

    return ToF


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
