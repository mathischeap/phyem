# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement

from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k2 import ___rc222_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221i_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221o_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k0 import ___rc220_msepy_quadrilateral___


class MseHttGreatMeshOrthogonalRectangleElement(MseHttGreatMeshBaseElement):
    r"""
    _________________________________> y
    |  0        face #0       2
    |  -----------------------
    |  |                     |
    |  |                     |
    |  | face #2             |face #3
    |  |                     |
    |  -----------------------
    v  1      face #1        3
    x

    The labels in _map refers to the for nodes in such a sequence.

    For example, _map = [87, 44, 156, 7561], then it is
    _________________________________> y
    |  87                    156
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  44                    7561
    x

    """

    def __init__(self, element_index, parameters, _map):
        r""""""
        origin_x, origin_y = parameters['origin']
        delta_x, delta_y = parameters['delta']
        self._metric_signature = f"OR:x%.5f" % delta_x + "y%.5f" % delta_y
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = MseHttGreatMeshOrthogonalRectangleElementCooTrans(self, origin_x, origin_y, delta_x, delta_y)

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
        r""""""
        return 'orthogonal rectangle'

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        origin_x, origin_y = parameters['origin']
        delta_x, delta_y = parameters['delta']
        return np.array([origin_x + delta_x / 2, origin_y + delta_y/2])

    def __repr__(self):
        r""""""
        super_repr = super().__repr__().split('object')[1]
        return f"<Orthogonal Rectangle element indexed:{self._index}" + super_repr

    @property
    def metric_signature(self):
        r""""""
        return self._metric_signature

    def _generate_outline_data(self, ddf=None, internal_grid=0):
        r""""""
        linspace = np.array([-1, 1])
        ones = np.ones_like(linspace)

        line_dict = {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: self.ct.mapping(-ones, linspace),   # face #0
            1: self.ct.mapping(ones, linspace),    # face #1
            2: self.ct.mapping(linspace, -ones),   # face #2
            3: self.ct.mapping(linspace, ones),    # face #3
        }

        if internal_grid == 0:
            return line_dict
        else:
            raise NotImplementedError()

    @classmethod
    def face_setting(cls):
        r"""To show the nodes of faces and the positive direction."""
        return {
            0: (0, 2),   # face #0 is from node 0 -> node 2  (positive direction), North
            1: (1, 3),   # face #1 is from node 1 -> node 3  (positive direction), South
            2: (0, 1),   # face #2 is from node 0 -> node 1  (positive direction), West
            3: (2, 3),   # face #3 is from node 2 -> node 3  (positive direction), East
        }

    @property
    def faces(self):
        r"""The faces of this element."""
        if self._faces is None:
            self._faces = MseHttGreatMeshOrthogonalRectangleElementFaces(self)
        return self._faces

    @property
    def edges(self):
        r""""""
        raise Exception(f"orthogonal_rectangle element has no edges.")

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

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"orthogonal_rectangle element has no edges.")

    # @classmethod
    # def _form_face_dof_direction_topology(cls):
    #     m2n2k1_outer = {
    #         0: '-',   # on the x- faces, material leave the element is negative.
    #         1: '+',   # on the x+ faces, material leave the element is positive.
    #         2: '-',   # on the y- faces, material leave the element is negative.
    #         3: '+',   # on the y+ faces, material leave the element is positive.
    #     }
    #
    #     m2n2k1_inner = {
    #         0: '-',  # on the x- faces, positive direction is from 2 to 0, i.e., reversed
    #         1: '+',  # on the x+ faces, positive direction is from 1 to 3
    #         2: '+',  # on the y- faces, positive direction is from 0 to 1
    #         3: '-',  # on the y+ faces, positive direction is from 3 to 2, i.e., reversed
    #     }
    #
    #     return {'m2n2k1_outer': m2n2k1_outer, 'm2n2k1_inner': m2n2k1_inner}

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
        if indicator == 'm2n2k2':  # must be Lambda
            dtype = '2d-scalar'
            rc = ___rc222_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k1_outer':   # must be Lambda
            dtype = '2d-vector'
            rc = ___rc221o_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k1_inner':   # must be Lambda
            dtype = '2d-vector'
            rc = ___rc221i_msepy_quadrilateral___(self, degree, element_cochain, linspace, linspace, ravel=False)
        elif indicator == 'm2n2k0':  # must be Lambda
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


# ============ ELEMENT CT =====================================================================================
class MseHttGreatMeshOrthogonalRectangleElementCooTrans(Frozen):
    r"""No need to use the standard CT form."""
    def __init__(self, element, origin_x, origin_y, delta_x, delta_y):
        r""""""
        self._element = element
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._ratio_x = delta_x / 2
        self._ratio_y = delta_y / 2
        self._ct_helper = ___ct_helper_parser___(self._ratio_x, self._ratio_y)
        self._freeze()

    def __repr__(self):
        r""""""
        return f"<CT of {self._element.__repr__()}>"

    def mapping(self, xi, et):
        r""""""
        r = (xi + 1) * self._ratio_x
        s = (et + 1) * self._ratio_y
        x = self._origin_x + r
        y = self._origin_y + s
        return x, y

    def Jacobian_matrix(self, xi, et):
        r""""""
        return self._ct_helper.Jacobian_matrix(xi, et)

    def Jacobian(self, xi, et):
        r""""""
        return self._ct_helper.Jacobian(xi, et)

    def inverse_Jacobian_matrix(self, xi, et):
        r""""""
        return self._ct_helper.inverse_Jacobian_matrix(xi, et)

    def metric(self, xi, et):
        r""""""
        return self._ct_helper.metric(xi, et)

    def inverse_Jacobian(self, xi, et):
        r""""""
        return self._ct_helper.inverse_Jacobian(xi, et)

    def metric_matrix(self, xi, et):
        r""""""
        return self._ct_helper.metric_matrix(xi, et)

    def inverse_metric_matrix(self, xi, et):
        r""""""
        return self._ct_helper.inverse_metric_matrix(xi, et)


_cache_ct_helper_pool_ = {}


def ___ct_helper_parser___(ratio_x, ratio_y):
    r""""""
    ratio_x = round(ratio_x, 8)
    ratio_y = round(ratio_y, 8)
    key = (ratio_x, ratio_y)
    if key in _cache_ct_helper_pool_:
        pass
    else:
        _cache_ct_helper_pool_[key] = ___OrthogonalRectangle___(ratio_x, ratio_y)
    return _cache_ct_helper_pool_[key]


# noinspection PyUnusedLocal
class ___OrthogonalRectangle___(Frozen):
    r""""""
    def __init__(self, ratio_x, ratio_y):
        r""""""
        self._Jacobian_constant = ratio_x * ratio_y
        self._metric_constant = self._Jacobian_constant ** 2

        reciprocalJacobian_constant = 1 / self._Jacobian_constant
        inverse_Jacobian_matrix_00 = reciprocalJacobian_constant * ratio_y
        inverse_Jacobian_matrix_11 = reciprocalJacobian_constant * ratio_x

        self._inverse_Jacobian_constant = inverse_Jacobian_matrix_00 * inverse_Jacobian_matrix_11

        self._Jacobian_matrix = (
            (ratio_x, 0),
            (0, ratio_y)
        )
        self._iJM = (
            (inverse_Jacobian_matrix_00, 0),
            (0, inverse_Jacobian_matrix_11)
        )

        self._mm = (
            [ratio_x ** 2, 0],
            [0, ratio_y ** 2]
        )

        self._imm = (
            [inverse_Jacobian_matrix_00 ** 2, 0],
            [0, inverse_Jacobian_matrix_11 ** 2]
        )

        self._freeze()

    def Jacobian_matrix(self, xi, et):
        r""""""
        return self._Jacobian_matrix

    def Jacobian(self, xi, et):
        r""""""
        return self._Jacobian_constant

    def inverse_Jacobian_matrix(self, xi, et):
        r""""""
        return self._iJM

    def metric(self, xi, et):
        r""""""
        return self._metric_constant

    def inverse_Jacobian(self, xi, et):
        r""""""
        return self._inverse_Jacobian_constant

    def metric_matrix(self, xi, et):
        r""""""
        return self._mm

    def inverse_metric_matrix(self, xi, et):
        r""""""
        return self._imm


# ============ FACES =====================================================================================

class MseHttGreatMeshOrthogonalRectangleElementFaces(Frozen):
    r""""""
    def __init__(self, element):
        r""""""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        r"""0, 1, 2, 3.

        _________________________________> y
        |           face #0
        |  -----------------------
        |  |                     |
        |  |                     |
        |  | face #2             |face #3
        |  |                     |
        |  -----------------------
        v         face #1
        x

        """
        assert face_id in range(4), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = MseHttGreatMeshOrthogonalRectangleElementFace(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        r""""""
        return f"<Faces of {self._element}>"


class MseHttGreatMeshOrthogonalRectangleElementFace(Frozen):
    r""""""
    def __init__(self, element, face_id):
        r""""""
        self._element = element
        self._id = face_id
        self._ct = MseHttGreatMeshOrthogonalRectangleElementFaceCT(self)
        self._freeze()

    def __repr__(self):
        r""""""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        r"""Coordinate transformation of this face."""
        return self._ct


from phyem.msehtt.static.mesh.great.elements.types.base import _FaceCoordinateTransformationBase


class MseHttGreatMeshOrthogonalRectangleElementFaceCT(_FaceCoordinateTransformationBase):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
        self._melt()
        fid = face._id
        self._axis, self._start_end = fid // 2, fid % 2
        self._freeze()

    def __repr__(self):
        r"""repr"""
        side = '+' if self._start_end == 1 else '-'
        return f"<Face CT of {side}side along {self._axis}axis of {self._element}>"

    def mapping(self, xi):
        r""""""
        m, n = self._axis, self._start_end
        ones = np.ones_like(xi)
        if m == 0:  # x-direction
            if n == 0:  # x-
                return self._element.ct.mapping(-ones, xi)
            elif n == 1:  # x+
                return self._element.ct.mapping(ones, xi)
            else:
                raise Exception()
        elif m == 1:  # y-direction
            if n == 0:  # y-
                return self._element.ct.mapping(xi, -ones)
            elif n == 1:  # y+
                return self._element.ct.mapping(xi, ones)
            else:
                raise Exception()
        else:
            raise Exception()

    def Jacobian_matrix(self, xi):
        r""""""
        m, n = self._axis, self._start_end

        ones = np.ones_like(xi)
        if m == 0:  # x-direction
            if n == 0:  # x-
                JM = self._element.ct.Jacobian_matrix(-ones, xi)
            elif n == 1:  # x+
                JM = self._element.ct.Jacobian_matrix(ones, xi)
            else:
                raise Exception()

            return JM[0][1], JM[1][1]

        elif m == 1:  # y-direction
            if n == 0:  # y-
                JM = self._element.ct.Jacobian_matrix(xi, -ones)
            elif n == 1:  # y+
                JM = self._element.ct.Jacobian_matrix(xi, ones)
            else:
                raise Exception()

            return JM[0][0], JM[1][0]

        else:
            raise Exception()

    def outward_unit_normal_vector(self, xi):
        r"""The outward unit norm vector (vec{n})."""
        JM = self.Jacobian_matrix(xi)
        x, y = JM
        m, n = self._axis, self._start_end
        if m == 0 and n == 0:
            vx, vy = -y, x
        elif m == 1 and n == 1:
            vx, vy = -y, x
        else:
            vx, vy = y, -x
        magnitude = np.sqrt(vx**2 + vy**2)
        return vx / magnitude, vy / magnitude

    def is_plane(self):
        r""""""
        return True
