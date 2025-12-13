# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen


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

    def __eq__(self, other):
        r""""""
        if other.__class__ is self.__class__:
            if (abs(self.x - other.x) < 1e-8 and
                    abs(self.y - other.y) < 1e-8 and
                    abs(self.z - other.z) < 1e-8):
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
        return "%.6f" % round(self._x, 6) + "P%.6f" % round(self._y, 6)

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

        In this case, since I am a 3d point, we just need to check if this point presents in `geometries`.

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
        self._length = None
        self._freeze()

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def length(self):
        if self._length is None:
            self._length = distance3(self.A, self.B)
        return self._length

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

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Return True iff `geo` is also a StraightSegment3 and it is a subset of self."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            # print(time(), flush=True)
            if self == geo:
                return True
            else:
                if whether_point_on_straight_segment(geo.A, self):
                    return whether_point_on_straight_segment(geo.B, self)
                else:
                    return False


class OrthogonalSegment3(Frozen):
    r""""""
    def __init__(self, axis, origin, length):
        r""""""
        assert origin.__class__ is Point3
        assert isinstance(length, (int, float))
        assert abs(length) > 1e-4, f"length too small."
        self._axis = axis
        self._length = length
        x, y, z = origin._x, origin._y, origin._z
        if self._axis == 'x':
            self._bounds = [x, x+length]
            self._line = (y, z)  # I am on this line
        elif self._axis == 'y':
            self._bounds = [y, y+length]
            self._line = (z, x)  # I am on this line
        elif self._axis == 'z':
            self._bounds = [z, z+length]
            self._line = (x, y)  # I am on this line
        else:
            raise Exception
        self._bounds.sort()
        self._freeze()

    @property
    def length(self):
        return self._length

    def __eq__(self, other):
        r""""""
        if isinstance(other, self.__class__):
            if self._axis != other._axis:
                return False
            else:
                sc0, sc1 = self._line
                os0, os1 = other._line
                if abs(sc0 - os0) > 1e-8 or abs(sc1 - os1) > 1e-8:
                    return False
                else:

                    sb0, sb1 = self._bounds
                    ob0, ob1 = other._bounds

                    if abs(sb0 - ob0) > 1e-8 or abs(ob1 - sb1) > 1e-8:
                        return False
                    else:
                        return True
        else:
            raise NotImplementedError()

    def ___possible_subset_group_key___(self):
        r"""Return a key-string, and other objects of the same key-string can be a subset
        of me. Or the other way around.
        """
        return rf"S{self._axis}" + '%.5f' % round(self._line[0], 5) + '-%.5f' % round(self._line[1], 5)

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Return True iff `geo` is also a StraightSegment3 and it is a subset of self."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            if self._axis != geo._axis:
                return False
            else:
                sc0, sc1 = self._line
                os0, os1 = geo._line
                if abs(sc0 - os0) > 1e-8 or abs(sc1 - os1) > 1e-8:
                    return False
                else:

                    sb0, sb1 = self._bounds
                    ob0, ob1 = geo._bounds

                    if (sb0 - ob0) > 1e-8 or (ob1 - sb1) > 1e-8:
                        return False
                    else:
                        return True

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
            assert geo.__class__ is self.__class__
            if geo._axis == self._axis:
                parallel_geometries.append(geo)
            else:
                pass

        possible_geometries = []
        for geo in parallel_geometries:
            if self.whether_contain__input__as_a_part_of_same_type(geo):
                possible_geometries.append(geo)
            else:
                pass

        # SLB, SUP = self._bounds
        #
        # current_LB = [round(SLB, 8), ]
        # while 1:
        #     new_LB = []
        #     for geo in possible_geometries:
        #         lb, up = geo._bounds
        #         if round(lb, 8) in current_LB:
        #             new_LB.append(up)
        #         else:
        #             pass
        #     if len(new_LB) == 0:
        #         return False
        #     elif round(SUP, 8) in new_LB:
        #         return True
        #     else:
        #         current_LB = new_LB

        # !!!!!!!!!! we assume that all geometries in possible_geometries are separate!!!
        # !!!!!!!!!! SO we do no more checks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if abs(self.length - sum([_.length for _ in possible_geometries])) < 1e-8:
            return True
        else:
            return False


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


class PerpRectangle(Frozen):
    r""""""
    def __init__(self, perp_axis, origin, delta):
        r"""For example,

        if perp_axis = 'x', origin = (x0, y0, z0), delta = (0, dy, dz)
        if perp_axis = 'y', origin = (x0, y0, z0), delta = (dx, 0, dz)
        if perp_axis = 'z', origin = (x0, y0, z0), delta = (dx, dy, 0)

        """
        if perp_axis in ('x', 'y', 'z'):
            pass
        else:
            raise Exception(f"perp_axis={perp_axis} is wrong, should be among 'x', 'y' and 'z'.")

        assert len(origin) == 3 and all([isinstance(_, (int, float))for _ in origin])
        assert len(delta) == 3 and all([isinstance(_, (int, float))for _ in delta])

        dx, dy, dz = delta
        x0, y0, z0 = origin
        self._perp_axis = perp_axis
        self._center = (x0 + dx / 2, y0 + dy / 2, z0 + dz / 2)

        if perp_axis == 'x':
            assert dx == 0
            assert abs(dy) > 1e-4 and abs(dz) > 1e-4
            self._face = x0
            self._bounds0 = [y0, y0 + dy]
            self._bounds1 = [z0, z0 + dz]
        elif perp_axis == 'y':
            assert dy == 0
            assert abs(dx) > 1e-4 and abs(dz) > 1e-4
            self._face = y0
            self._bounds0 = [z0, z0 + dz]
            self._bounds1 = [x0, x0 + dx]
        elif perp_axis == 'z':
            assert dz == 0
            assert abs(dx) > 1e-4 and abs(dy) > 1e-4
            self._face = z0
            self._bounds0 = [x0, x0 + dx]
            self._bounds1 = [y0, y0 + dy]
        else:
            raise Exception()

        self._bounds0.sort()
        self._bounds1.sort()

        self._area = np.prod([self._bounds0[1] - self._bounds0[0], self._bounds1[1] - self._bounds1[0]])
        self._freeze()

    @property
    def area(self):
        return self._area

    def __eq__(self, other):
        r""""""
        if isinstance(other, self.__class__):
            if self._perp_axis != other._perp_axis or abs(self._face - other._face) > 1e-8:
                return False
            else:
                sB0_L, sB0_U = self._bounds0
                sB1_L, sB1_U = self._bounds1
                oB0_L, oB0_U = other._bounds0
                oB1_L, oB1_U = other._bounds1
                if abs(sB0_L - oB0_L) > 1e-8 or abs(oB0_U - sB0_U) > 1e-8:
                    return False
                elif abs(sB1_L - oB1_L) > 1e-8 or abs(oB1_U - sB1_U) > 1e-8:
                    return False
                else:
                    return True

        else:
            raise NotImplementedError()

    def ___possible_subset_group_key___(self):
        r"""Return a key-string, and other objects of the same key-string can be a subset
        of me. Or the other way around.
        """
        return rf"pR{self._perp_axis}" + '%.6f' % round(self._face, 6)

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Return True iff `geo` is also a perp-rectangle and it is a subset of self."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            if self._perp_axis != geo._perp_axis or abs(self._face - geo._face) > 1e-8:
                return False
            else:
                sB0_L, sB0_U = self._bounds0
                sB1_L, sB1_U = self._bounds1
                oB0_L, oB0_U = geo._bounds0
                oB1_L, oB1_U = geo._bounds1
                if sB0_L - oB0_L > 1e-8 or oB0_U - sB0_U > 1e-8:
                    return False
                elif sB1_L - oB1_L > 1e-8 or oB1_U - sB1_U > 1e-8:
                    return False
                else:
                    return True

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
            assert isinstance(geo, self.__class__)
            if geo._perp_axis == self._perp_axis and abs(self._face - geo._face) < 1e-8:
                parallel_geometries.append(geo)
            else:
                pass

        possible_geometries = []
        for geo in parallel_geometries:
            if self.whether_contain__input__as_a_part_of_same_type(geo):
                possible_geometries.append(geo)
            else:
                pass

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!! we assume that all geometries in possible_geometries are separate!!!
        # !!!!!!!!!! SO we do no more checks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if abs(self.area - sum([_.area for _ in possible_geometries])) < 1e-7:
            return True
        else:
            return False


class OrthogonalHexahedron(Frozen):
    r""""""
    def __init__(self, origin, delta):
        r""""""
        assert len(origin) == 3 and all([isinstance(_, (int, float))for _ in origin])
        assert len(delta) == 3 and all([isinstance(_, (int, float))for _ in delta])
        dx, dy, dz = delta
        x0, y0, z0 = origin
        self._center = (x0 + dx / 2, y0 + dy / 2, z0 + dz / 2)

        assert abs(dx) > 1e-4 and abs(dy) > 1e-4 and abs(dz) > 1e-4

        self._d_xyz = (abs(dx), abs(dy), abs(dz))

        self._bounds0 = [x0, x0 + dy]
        self._bounds1 = [y0, y0 + dy]
        self._bounds2 = [z0, z0 + dz]

        self._bounds0.sort()
        self._bounds1.sort()
        self._bounds2.sort()
        self._volume = np.prod(
            [
                self._bounds0[1] - self._bounds0[0],
                self._bounds1[1] - self._bounds1[0],
                self._bounds2[1] - self._bounds2[0]
            ]
        )
        self._freeze()

    @property
    def volume(self):
        r""""""
        return self._volume

    def __eq__(self, other):
        r""""""
        if isinstance(other, self.__class__):
            x0, y0, z0 = self._center
            x1, y1, z1 = other._center
            dx0, dy0, dz0 = self._d_xyz
            dx1, dy1, dz1 = other._d_xyz
            if (abs(x0 - x1) < 1e-8 and
                    abs(y0 - y1) < 1e-8 and
                    abs(z0 - z1) < 1e-8 and
                    abs(dx0 - dx1) < 1e-8 and
                    abs(dy0 - dy1) < 1e-8 and
                    abs(dz0 - dz1) < 1e-8):
                return True
            else:
                return False
        else:
            raise NotImplementedError()

    def whether_contain__input__as_a_part_of_same_type(self, geo):
        r"""Return True iff `geo` is also a Orthogonal-hexa and it is a subset of self."""
        if geo.__class__ is not self.__class__:
            return False
        else:
            sB0_L, sB0_U = self._bounds0
            sB1_L, sB1_U = self._bounds1
            sB2_L, sB2_U = self._bounds2
            oB0_L, oB0_U = geo._bounds0
            oB1_L, oB1_U = geo._bounds1
            oB2_L, oB2_U = geo._bounds2
            if sB0_L - oB0_L > 1e-8 or oB0_U - sB0_U > 1e-8:
                return False
            elif sB1_L - oB1_L > 1e-8 or oB1_U - sB1_U > 1e-8:
                return False
            elif sB2_L - oB2_L > 1e-8 or oB2_U - sB2_U > 1e-8:
                return False
            else:
                return True

    def is_equal_to_a_union_of_a_part_of_separate_geometries_of_same_type(self, geometries):
        r"""check whether this geo is equal to the Union of some of `geometries`.

        `separate` means intersection of geometries can only be None or a geo of a lower dimension than mine.

        Remember, it must be `equal`, not be a subset. So, we must could find some geometries from `geometries`
        whose union is exactly equal to self.

        `geometries` must be a list or tuple of geometries of the same type as self.

        We will skip the non-parallel segments in geometries since they have no effect at all. So, even some
        of them are not separate, we do not care.
        """

        possible_geometries = []
        for geo in geometries:
            assert isinstance(geo, self.__class__)
            if self.whether_contain__input__as_a_part_of_same_type(geo):
                possible_geometries.append(geo)
            else:
                pass

        # !!!!!!!!!! we assume that all geometries in possible_geometries are separate!!!
        # !!!!!!!!!! SO we do no more checks!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        if abs(self.volume - sum([_.volume for _ in possible_geometries])) < 1e-7:
            return True
        else:
            return False


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
    dAB = segment.length
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
