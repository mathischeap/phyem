# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen

from tools.functions.space._2d.angle import angle
from tools.functions.space._2d.angles_of_triangle import angles_of_triangle
from tools.functions.space._2d.distance import distance

from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement

from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k2 import ___rc222_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221i_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221o_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m2n2k0 import ___rc220_msepy_quadrilateral___


___A___ = np.array([
    [1, 0, 0, 0],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [1, 0, 1, 0],
])

___invA___ = np.linalg.inv(___A___)


class Vtu9Quad(MseHttGreatMeshBaseElement):
    r"""
    _________________________________>  eta
    |  0        face #0       3
    |  -----------------------
    |  |                     |
    |  |                     |
    |  | face #2             |face #3
    |  |                     |
    |  -----------------------
    v  1      face #1        2

    xi

    The labels in _map refers to the for nodes in such a sequence.

    For example, _map = [87, 44, 7561, 156], then it is
    _________________________________> eta
    |  87                    156
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  44                    7561
    xi

    And the real number inherits the numbering.

    """

    def __init__(self, element_index, parameters, _map):
        r""""""
        a0, b, c0 = angles_of_triangle(parameters[0], parameters[1], parameters[2])   # in degree
        a1, c1, _ = angles_of_triangle(parameters[0], parameters[2], parameters[3])   # in degree
        a = a0 + a1
        c = c0 + c1

        d = angle(parameters[0], parameters[2])
        dist = distance(parameters[0], parameters[2])
        self._metric_signature = f"9:a%.2f" % a + "b%.2f" % b + "c%.2f" % c + "d%.3f" % d + "dis%.5f" % dist

        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = Vtu9Quad_CT(self)

    @classmethod
    def m(cls):
        r"""the dimensions of the space"""
        return 2

    @classmethod
    def n(cls):
        r"""the dimensions of the element"""
        return 2

    @classmethod
    def _etype(cls):
        r"""vtu cell type 9: quad"""
        return 9

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        return np.sum(np.array(parameters), axis=0) / 4

    @classmethod
    def _find_mapping_(cls, parameters, xi, et):
        r""""""
        vec = np.array(parameters)

        x = vec[:, 0]
        y = vec[:, 1]

        alpha = ___invA___ @ x
        beta = ___invA___ @ y

        a1, a2, a3, a4 = alpha
        b1, b2, b3, b4 = beta
        # to the reference square

        r = (xi + 1) * 0.5
        s = (et + 1) * 0.5
        # to the physical quad
        x = a1 + a2 * r + a3 * s + a4 * r * s
        y = b1 + b2 * r + b3 * s + b4 * r * s
        return x, y

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<VTU-9 element indexed:{self._index}" + super_repr

    @property
    def metric_signature(self):
        r""""""
        return self._metric_signature

    def _generate_outline_data(self, ddf=None, internal_grid=0):
        r""""""
        x0, y0 = self._parameters[0]
        x1, y1 = self._parameters[1]
        x2, y2 = self._parameters[2]
        x3, y3 = self._parameters[3]
        data_dict = {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            2: ([x0, x1], [y0, y1]),
            1: ([x1, x2], [y1, y2]),
            3: ([x2, x3], [y2, y3]),
            0: ([x3, x0], [y3, y0]),
        }

        if internal_grid == 0:
            return data_dict
        else:
            ONE = np.array([-1, 1])
            LinSpace = np.linspace(-1, 1, internal_grid+2)[1:-1]
            internal_line_number = 0
            for i in LinSpace:
                Span = i * np.ones(2)
                x, y = self.ct.mapping(Span, ONE)
                data_dict[f"internal_line_x_{internal_line_number}"] = (x, y)
                x, y = self.ct.mapping(ONE, Span)
                data_dict[f"internal_line_y_{internal_line_number}"] = (x, y)
                internal_line_number += 1

            return data_dict

    def _find_element_quality(self):
        r"""Return a factor indicating the quality of the elements.

        When the factor is 0: the element is worst.
        When the factor is 1: the element is best.

        Returns
        -------
        quality: float
            In [0, 1]. 0: Worst; 1: Best.

        """
        parameters = self.parameters
        a0, b, c0 = angles_of_triangle(parameters[0], parameters[1], parameters[2])   # in degree
        a1, c1, d = angles_of_triangle(parameters[0], parameters[2], parameters[3])   # in degree
        a = a0 + a1
        c = c0 + c1
        quality = np.array([a - 90, b - 90, c - 90, d - 90])
        quality = np.sqrt(np.sum(quality.dot(quality)))
        quality /= 180
        quality = 1 - quality
        if quality < 1e-8:
            quality = 0
        elif quality > 9.9999:
            quality = 1
        else:
            pass
        return quality

    @classmethod
    def face_setting(cls):
        r"""To show the nodes of faces and the positive direction."""
        return {
            0: (0, 3),   # face #0 is from node 0 -> node 3  (positive direction)
            1: (1, 2),   # face #1 is from node 1 -> node 2  (positive direction)
            2: (0, 1),   # face #2 is from node 0 -> node 1  (positive direction)
            3: (3, 2),   # face #3 is from node 3 -> node 2  (positive direction)
        }

    @property
    def faces(self):
        r"""The faces of this element."""
        if self._faces is None:
            self._faces = Quad_Faces(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([-1, 1, 0, 0])
        y = np.array([0, 0, -1, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (x[0], y[0]),
            1: r"%.7f-%.7f" % (x[1], y[1]),
            2: r"%.7f-%.7f" % (x[2], y[2]),
            3: r"%.7f-%.7f" % (x[3], y[3]),
        }

    @property
    def edges(self):
        r""""""
        raise Exception(f"vtu quad element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"vtu quad element has no edges.")

    def _generate_element_vtk_data_(self, xi, et):
        r""""""
        assert xi.ndim == et.ndim == 1
        sx, sy = xi.size, et.size
        meshgrid = np.meshgrid(xi, et, indexing='ij')
        X, Y = self.ct.mapping(*meshgrid)
        coo_dict = {}
        for j in range(sy):
            for i in range(sx):
                x, y = X[i, j], Y[i, j]
                key = f"%.7f-%.7f" % (x, y)
                coo_dict[key] = (x, y)
        cell_list = list()
        for j in range(sy - 1):
            for i in range(sx - 1):
                cell_list.append((
                    [
                        f"%.7f-%.7f" % (X[i, j], Y[i, j]),
                        f"%.7f-%.7f" % (X[i+1, j], Y[i+1, j]),
                        f"%.7f-%.7f" % (X[i, j+1], Y[i, j+1]),
                        f"%.7f-%.7f" % (X[i+1, j+1], Y[i+1, j+1]),
                    ], 4, 8)  # for this element, VTK_PIXEL cell (No. 8)!
                )
        return coo_dict, cell_list

    def _generate_vtk_data_for_form(self, indicator, element_cochain, degree, data_density):
        r""""""
        linspace = np.linspace(-1, 1, data_density)
        if indicator == 'm2n2k2':           # must be Lambda
            dtype = '2d-scalar'
            rc = ___rc222_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k1_outer':   # must be Lambda
            dtype = '2d-vector'
            rc = ___rc221o_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k1_inner':   # must be Lambda
            dtype = '2d-vector'
            rc = ___rc221i_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k0':         # must be Lambda
            dtype = '2d-scalar'
            rc = ___rc220_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        else:
            raise NotImplementedError()

        data_dict = {}

        if dtype == '2d-scalar':
            X, Y, V = rc
            for i in range(data_density):
                for j in range(data_density):
                    x = X[i][j]
                    y = Y[i][j]
                    v = V[i][j]
                    key = "%.7f-%.7f" % (x, y)
                    data_dict[key] = (x, y, v)

        elif dtype == '2d-vector':
            X, Y, U, V = rc
            for i in range(data_density):
                for j in range(data_density):
                    x = X[i][j]
                    y = Y[i][j]
                    u = U[i][j]
                    v = V[i][j]
                    key = "%.7f-%.7f" % (x, y)
                    data_dict[key] = (x, y, u, v)
        else:
            raise NotImplementedError()

        cell_list = list()
        for i in range(data_density - 1):
            for j in range(data_density - 1):
                cell_list.append((
                    [
                        "%.7f-%.7f" % (X[i][j], Y[i][j]),
                        "%.7f-%.7f" % (X[i + 1][j], Y[i + 1][j]),
                        "%.7f-%.7f" % (X[i + 1][j + 1], Y[i + 1][j + 1]),
                        "%.7f-%.7f" % (X[i][j + 1], Y[i][j + 1]),
                    ], 4, 9)
                )

        return data_dict, cell_list, dtype


from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans
# from msepy.manifold.predefined._helpers import _Transfinite2


class Vtu9Quad_CT(MseHttGreatMeshBaseElementCooTrans):
    r""""""

    # ----- CT implementation #1: quad transformation ------------------------------------------
    def __init__(self, vtu9e):
        r""""""
        super().__init__(vtu9e, vtu9e.metric_signature)

        self._melt()

        vec = np.array(vtu9e.parameters)

        x = vec[:, 0]
        y = vec[:, 1]

        alpha = ___invA___ @ x
        beta = ___invA___ @ y

        self._a1, self._a2, self._a3, self._a4 = alpha
        self._b1, self._b2, self._b3, self._b4 = beta

        self._freeze()

    def mapping(self, xi, et):
        r""""""
        # to the reference square
        r = (xi + 1) * 0.5
        s = (et + 1) * 0.5
        # to the physical quad
        x = self._a1 + self._a2 * r + self._a3 * s + self._a4 * r * s
        y = self._b1 + self._b2 * r + self._b3 * s + self._b4 * r * s
        return x, y

    def ___Jacobian_matrix___(self, xi, et):
        r""""""
        r = (xi + 1) * 0.5
        s = (et + 1) * 0.5

        dx_dr = self._a2 + self._a4 * s
        dx_ds = self._a3 + self._a4 * r

        dy_dr = self._b2 + self._b4 * s
        dy_ds = self._b3 + self._b4 * r

        dx_dxi = dx_dr * 0.5
        dx_det = dx_ds * 0.5

        dy_dxi = dy_dr * 0.5
        dy_det = dy_ds * 0.5

        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det]
        )

    # # ----- CT implementation #2: transfinite mapping -----------------------------------------
    # def __init__(self, vtu9e):
    #     """"""
    #     super().__init__(vtu9e, vtu9e.metric_signature)
    #
    #     self._melt()
    #
    #     vec = np.array(vtu9e.parameters)
    #
    #     # x = vec[:, 0]
    #     # y = vec[:, 1]
    #     #
    #     # x0, x1, x2, x3 = x
    #     # y0, y1, y2, y3 = y
    #
    #     geo_x0 = ['straight line', [vec[0], vec[3]]]
    #     geo_x1 = ['straight line', [vec[1], vec[2]]]
    #     geo_y0 = ['straight line', [vec[0], vec[1]]]
    #     geo_y1 = ['straight line', [vec[3], vec[2]]]
    #     self._tf = _Transfinite2(geo_x0, geo_x1, geo_y0, geo_y1)
    #     self._freeze()
    #
    # def mapping(self, xi, et):
    #     """"""
    #     # to the reference square
    #     r = (xi + 1) * 0.5
    #     s = (et + 1) * 0.5
    #     return self._tf.mapping(r, s)
    #
    # def ___Jacobian_matrix___(self, xi, et):
    #     """"""
    #     r = (xi + 1) * 0.5
    #     s = (et + 1) * 0.5
    #
    #     dx, dy = self._tf.Jacobian_matrix(r, s)
    #     dx_dr, dx_ds = dx
    #     dy_dr, dy_ds = dy
    #
    #     dx_dxi = dx_dr * 0.5
    #     dx_det = dx_ds * 0.5
    #
    #     dy_dxi = dy_dr * 0.5
    #     dy_det = dy_ds * 0.5
    #
    #     return (
    #         [dx_dxi, dx_det],
    #         [dy_dxi, dy_det]
    #     )


# ============ FACES ============================================================================================


class Quad_Faces(Frozen):
    r""""""
    def __init__(self, element):
        r""""""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        r"""0, 1, 2, 3.

       _________________________________> eta
        |  0        face #0       3
        |  -----------------------
        |  |                     |
        |  |         (ref)       |
        |  | face #2             |face #3
        |  |                     |
        |  -----------------------
        v  1      face #1        2
        xi

        """
        assert face_id in range(4), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = Quad_Face(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        r""""""
        return f"<Faces of {self._element}>"


class Quad_Face(Frozen):
    r""""""
    def __init__(self, element, face_id):
        r""""""
        self._element = element
        self._id = face_id
        self._ct = Quad_Face_CT(self)
        self._freeze()

    def __repr__(self):
        r""""""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        r"""Coordinate transformation of this face."""
        return self._ct


from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import \
    MseHttGreatMeshOrthogonalRectangleElementFaceCT


class Quad_Face_CT(
    MseHttGreatMeshOrthogonalRectangleElementFaceCT
):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
