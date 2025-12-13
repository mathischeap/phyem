# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.quadrature import quadrature

from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k0 import local_numbering_Lambda__m2n2k0
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import local_numbering_Lambda__m2n2k1_inner
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k1 import local_numbering_Lambda__m2n2k1_outer
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m2n2k2 import local_numbering_Lambda__m2n2k2

from phyem.tools.miscellaneous.geometries.m2n2 import Point2
from phyem.tools.miscellaneous.geometries.m2n2 import StraightSegment2
from phyem.tools.miscellaneous.geometries.m2n2 import Polygon2

from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k0 import local_numbering_Lambda__m3n3k0
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k1 import local_numbering_Lambda__m3n3k1
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k2 import local_numbering_Lambda__m3n3k2
from phyem.msehtt.static.space.local_numbering.Lambda.ln_m3n3k3 import local_numbering_Lambda__m3n3k3
from phyem.tools.miscellaneous.geometries.m3n3 import Point3
from phyem.tools.miscellaneous.geometries.m3n3 import OrthogonalSegment3
from phyem.tools.miscellaneous.geometries.m3n3 import PerpRectangle
from phyem.tools.miscellaneous.geometries.m3n3 import OrthogonalHexahedron


def _find_geo_local_dof_m2n2k0_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m2n2k0-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        9,
        'orthogonal rectangle',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering = local_numbering_Lambda__m2n2k0(etype, p)
        i, j = np.where(local_numbering == local_dof_index)
        assert np.size(i) == np.size(j) == 1, f"We must only find one local numbering in an element."
        i = int(i[0])
        j = int(j[0])
        nodes_xi, nodes_et = quadrature(p, dtype).quad_nodes
        xi = nodes_xi[i]
        et = nodes_et[j]
        x, y = element.ct.mapping(xi, et)
        return Point2(x, y)
    else:
        raise NotImplementedError()


def _find_geo_local_dof_m2n2k1_inner_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m2n2k1-inner-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        9,
        'orthogonal rectangle',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering_dx, local_numbering_dy = local_numbering_Lambda__m2n2k1_inner(etype, p)
        I, J = np.where(local_numbering_dy == local_dof_index)
        K, L = np.where(local_numbering_dx == local_dof_index)
        if np.size(I) == np.size(J) == 1:
            assert np.size(K) == np.size(L) == 0
            i = int(I[0])
            j = int(J[0])
            edge = 'dy'
        else:
            assert np.size(K) == np.size(L) == 1
            i = int(K[0])
            j = int(L[0])
            edge = 'dx'

        nodes_x, nodes_y = quadrature(p, dtype).quad_nodes

        if edge == 'dy':
            xi_0 = xi_1 = nodes_x[i]
            et_0 = nodes_y[j]
            et_1 = nodes_y[j+1]
        elif edge == 'dx':
            xi_0 = nodes_x[i]
            xi_1 = nodes_x[i+1]
            et_0 = et_1 = nodes_y[j]
        else:
            raise Exception
        x, y = element.ct.mapping(np.array([xi_0, xi_1]), np.array([et_0, et_1]))
        x0, x1 = x
        y0, y1 = y
        point0 = Point2(x0, y0)
        point1 = Point2(x1, y1)
        return StraightSegment2(point0, point1)
    else:
        raise NotImplementedError()


def _find_geo_local_dof_m2n2k1_outer_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m2n2k1-outer-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        9,
        'orthogonal rectangle',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering_dy, local_numbering_dx = local_numbering_Lambda__m2n2k1_outer(etype, p)
        I, J = np.where(local_numbering_dy == local_dof_index)
        K, L = np.where(local_numbering_dx == local_dof_index)
        if np.size(I) == np.size(J) == 1:
            assert np.size(K) == np.size(L) == 0
            i = int(I[0])
            j = int(J[0])
            edge = 'dy'
        else:
            assert np.size(K) == np.size(L) == 1
            i = int(K[0])
            j = int(L[0])
            edge = 'dx'

        nodes_x, nodes_y = quadrature(p, dtype).quad_nodes

        if edge == 'dy':
            xi_0 = xi_1 = nodes_x[i]
            et_0 = nodes_y[j]
            et_1 = nodes_y[j+1]
        elif edge == 'dx':
            xi_0 = nodes_x[i]
            xi_1 = nodes_x[i+1]
            et_0 = et_1 = nodes_y[j]
        else:
            raise Exception
        x, y = element.ct.mapping(np.array([xi_0, xi_1]), np.array([et_0, et_1]))
        x0, x1 = x
        y0, y1 = y
        point0 = Point2(x0, y0)
        point1 = Point2(x1, y1)
        return StraightSegment2(point0, point1)
    else:
        raise NotImplementedError()


def _find_geo_local_dof_m2n2k2_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m2n2k2-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        9,
        'orthogonal rectangle',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering = local_numbering_Lambda__m2n2k2(etype, p)
        i, j = np.where(local_numbering == local_dof_index)
        assert np.size(i) == np.size(j) == 1, f"We must only find one local numbering in an element."
        i = int(i[0])
        j = int(j[0])
        nodes_xi, nodes_et = quadrature(p, dtype).quad_nodes
        xi_0 = nodes_xi[i]
        xi_1 = nodes_xi[i+1]
        et_0 = nodes_et[j]
        et_1 = nodes_et[j+1]
        x, y = element.ct.mapping(np.array([xi_0, xi_1]), np.array([et_0, et_1]))
        x0, x1 = x
        y0, y1 = y
        point0 = Point2(x0, y0)
        point1 = Point2(x1, y0)
        point2 = Point2(x1, y1)
        point3 = Point2(x0, y1)
        return Polygon2(point0, point1, point2, point3)

    else:
        raise NotImplementedError()


def _find_geo_local_dof_m3n3k0_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m3n3k0-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        11,
        'orthogonal hexahedron',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering = local_numbering_Lambda__m3n3k0(etype, p)
        i, j, k = np.where(local_numbering == local_dof_index)
        assert np.size(i) == np.size(j) == np.size(k) == 1, f"We must only find one local numbering in an element."
        i = int(i[0])
        j = int(j[0])
        k = int(k[0])
        nodes_xi, nodes_et, nodes_sg = quadrature(p, dtype).quad_nodes
        xi = nodes_xi[i]
        et = nodes_et[j]
        sg = nodes_sg[k]
        x, y, z = element.ct.mapping(xi, et, sg)
        return Point3(x, y, z)

    else:
        raise NotImplementedError(etype)


def _find_geo_local_dof_m3n3k1_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m3n3k1-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        11,
        'orthogonal hexahedron',
    ):
        p, dtype = element.degree_parser(degree)
        LN_dx, LN_dy, LN_dz = local_numbering_Lambda__m3n3k1(etype, p)
        I0, I1, I2 = np.where(LN_dx == local_dof_index)
        J0, J1, J2 = np.where(LN_dy == local_dof_index)
        K0, K1, K2 = np.where(LN_dz == local_dof_index)
        if np.size(I0) == np.size(I1) == np.size(I2) == 1:
            assert np.size(J0) == np.size(J1) == np.size(J2) == 0
            assert np.size(K0) == np.size(K1) == np.size(K2) == 0
            i = int(I0[0])
            j = int(I1[0])
            k = int(I2[0])
            edge = 'x'
        elif np.size(J0) == np.size(J1) == np.size(J2) == 1:
            i = int(J0[0])
            j = int(J1[0])
            k = int(J2[0])
            edge = 'y'
        elif np.size(K0) == np.size(K1) == np.size(K2) == 1:
            i = int(K0[0])
            j = int(K1[0])
            k = int(K2[0])
            edge = 'z'
        else:
            raise Exception()

        nodes_x, nodes_y, nodes_z = quadrature(p, dtype).quad_nodes

        if edge == 'x':
            xi_0 = nodes_x[i]
            xi_1 = nodes_x[i+1]
            et_0 = et_1 = nodes_y[j]
            sg_0 = sg_1 = nodes_z[k]
        elif edge == 'y':
            xi_0 = xi_1 = nodes_x[i]
            et_0 = nodes_y[j]
            et_1 = nodes_y[j+1]
            sg_0 = sg_1 = nodes_z[k]
        elif edge == 'z':
            xi_0 = xi_1 = nodes_x[i]
            et_0 = et_1 = nodes_y[j]
            sg_0 = nodes_z[k]
            sg_1 = nodes_z[k+1]
        else:
            raise Exception
        x, y, z = element.ct.mapping(
            np.array([xi_0, xi_1]),
            np.array([et_0, et_1]),
            np.array([sg_0, sg_1]),
        )
        x0, x1 = x
        y0, y1 = y
        z0, z1 = z
        point0 = Point3(x0, y0, z0)

        if edge == 'x':
            length = x1 - x0
        elif edge == 'y':
            length = y1 - y0
        elif edge == 'z':
            length = z1 - z0
        else:
            raise Exception

        return OrthogonalSegment3(edge, point0, length)
    else:
        raise NotImplementedError(etype)


def _find_geo_local_dof_m3n3k2_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m3n3k2-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        11,
        'orthogonal hexahedron',
    ):
        p, dtype = element.degree_parser(degree)
        LN_dydz, LN_dzdx, LN_dxdy = local_numbering_Lambda__m3n3k2(etype, p)
        I0, I1, I2 = np.where(LN_dydz == local_dof_index)
        J0, J1, J2 = np.where(LN_dzdx == local_dof_index)
        K0, K1, K2 = np.where(LN_dxdy == local_dof_index)
        if np.size(I0) == np.size(I1) == np.size(I2) == 1:
            assert np.size(J0) == np.size(J1) == np.size(J2) == 0
            assert np.size(K0) == np.size(K1) == np.size(K2) == 0
            i = int(I0[0])
            j = int(I1[0])
            k = int(I2[0])
            perp = 'x'
        elif np.size(J0) == np.size(J1) == np.size(J2) == 1:
            i = int(J0[0])
            j = int(J1[0])
            k = int(J2[0])
            perp = 'y'
        elif np.size(K0) == np.size(K1) == np.size(K2) == 1:
            i = int(K0[0])
            j = int(K1[0])
            k = int(K2[0])
            perp = 'z'
        else:
            raise Exception()

        nodes_x, nodes_y, nodes_z = quadrature(p, dtype).quad_nodes

        if perp == 'x':
            xi__ = nodes_x[i]
            et_0 = nodes_y[j]
            et_1 = nodes_y[j+1]
            sg_0 = nodes_z[k]
            sg_1 = nodes_z[k+1]

            x, y, z = element.ct.mapping(
                np.array([xi__, xi__, xi__]),
                np.array([et_0, et_1, et_0]),
                np.array([sg_0, sg_0, sg_1]),
            )

            origin = (x[0], y[0], z[0])
            dx = 0
            dy = y[1] - y[0]
            dz = z[2] - z[0]

        elif perp == 'y':
            xi_0 = nodes_x[i]
            xi_1 = nodes_x[i+1]
            et__ = nodes_y[j]
            sg_0 = nodes_z[k]
            sg_1 = nodes_z[k+1]

            x, y, z = element.ct.mapping(
                np.array([xi_0, xi_1, xi_0]),
                np.array([et__, et__, et__]),
                np.array([sg_0, sg_0, sg_1]),
            )

            origin = (x[0], y[0], z[0])
            dx = x[1] - x[0]
            dy = 0
            dz = z[2] - z[0]

        elif perp == 'z':
            xi_0 = nodes_x[i]
            xi_1 = nodes_x[i+1]
            et_0 = nodes_y[j]
            et_1 = nodes_y[j+1]
            sg__ = nodes_z[k]

            x, y, z = element.ct.mapping(
                np.array([xi_0, xi_1, xi_0]),
                np.array([et_0, et_0, et_1]),
                np.array([sg__, sg__, sg__]),
            )

            origin = (x[0], y[0], z[0])
            dx = x[1] - x[0]
            dy = y[2] - y[0]
            dz = 0

        else:
            raise Exception

        return PerpRectangle(perp, origin, (dx, dy, dz))

    else:
        raise NotImplementedError(etype)


def _find_geo_local_dof_m3n3k3_(degree, element, local_dof_index):
    r"""Return a geometric object for the local dof #local_dof_index of m3n3k3-space @ `degree` in `element`.
    """
    etype = element._etype()
    if etype in (
        11,
        'orthogonal hexahedron',
    ):
        p, dtype = element.degree_parser(degree)
        local_numbering = local_numbering_Lambda__m3n3k3(etype, p)
        i, j, k = np.where(local_numbering == local_dof_index)
        assert np.size(i) == np.size(j) == np.size(k) == 1, f"We must only find one local numbering in an element."
        i = int(i[0])
        j = int(j[0])
        k = int(k[0])
        nodes_xi, nodes_et, nodes_sg = quadrature(p, dtype).quad_nodes
        xi_0 = nodes_xi[i]
        xi_1 = nodes_xi[i+1]
        et_0 = nodes_et[j]
        et_1 = nodes_et[j+1]
        sg_0 = nodes_sg[k]
        sg_1 = nodes_sg[k+1]
        x, y, z = element.ct.mapping(
            np.array([xi_0, xi_1, xi_0, xi_0]),
            np.array([et_0, et_0, et_1, et_0]),
            np.array([sg_0, et_0, sg_0, sg_1]),
        )
        origin = (x[0], y[0], z[0])
        delta = (
            x[1] - x[0], y[2] - y[0], z[3] - z[0]
        )
        return OrthogonalHexahedron(origin, delta)

    else:
        raise NotImplementedError(etype)
