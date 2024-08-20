# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement

from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k0 import ___rc330_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k1 import ___rc331_orthogonal_hexahedron___
from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k2 import ___rc332_msepy_quadrilateral___
from msehtt.static.space.reconstruct.Lambda.Rc_m3n3k3 import ___rc333_msepy_quadrilateral___


class MseHttGreatMeshOrthogonalHexahedronElement(MseHttGreatMeshBaseElement):
    """
    Local node numbering:

    back-face: z- face

    _________________________________> y
    |  0                      2
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  1                     3
    x

    forward-face: z+ face

    _________________________________> y
    |  4                      6
    |  -----------------------
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  5                      7
    x


    """

    def __init__(self, element_index, parameters, _map):
        """"""
        origin_x, origin_y, origin_z = parameters['origin']
        delta_x, delta_y, delta_z = parameters['delta']
        self._metric_signature = f"OR:x%.5f" % delta_x + "y%.5f" % delta_y + "y%.5f" % delta_z
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._freeze()
        self._ct = MseHtt_GreatMesh_OrthogonalHexahedron_Element_CooTrans(
            self,
            origin_x, origin_y, origin_z,
            delta_x, delta_y, delta_z
        )

    @classmethod
    def m(cls):
        """the dimensions of the space"""
        return 3

    @classmethod
    def n(cls):
        """the dimensions of the element"""
        return 3

    @classmethod
    def _etype(cls):
        return 'orthogonal hexahedron'

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Orthogonal Hexahedron element indexed:{self._index}" + super_repr

    @property
    def metric_signature(self):
        """"""
        return self._metric_signature

    @classmethod
    def face_setting(cls):
        """To show the nodes of faces and the positive direction."""
        return {
            0: (0, 2, 4, 6),   # face #0: x- face, Upper
            1: (1, 3, 5, 7),   # face #1: x+ face, Down
            2: (0, 1, 4, 5),   # face #2: y- face, Left
            3: (2, 3, 6, 7),   # face #3: y+ face, Right
            4: (0, 1, 2, 3),   # face #2: z- face, Back
            5: (4, 5, 6, 7),   # face #3: z+ face, Front
        }

    @property
    def faces(self):
        """The faces of this element."""
        if self._faces is None:
            self._faces = MseHtt_GreatMesh_OrthogonalHexahedron_Element_Faces(self)
        return self._faces

    @classmethod
    def degree_parser(cls, degree):
        """"""
        if isinstance(degree, int):
            p = (degree, degree, degree)
            dtype = 'Lobatto'
        else:
            raise NotImplementedError()
        return p, dtype

    def _generate_outline_data(self, ddf=1):
        linspace = np.array([-1, 1, 1, -1, -1]), np.array([-1, -1, 1, 1, -1])
        ones = np.ones(5)

        return {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0, 0),
            0: self.ct.mapping(-ones, linspace[0], linspace[1]),   # face #0, x-
            1: self.ct.mapping(ones, linspace[0], linspace[1]),    # face #1, x+
            2: self.ct.mapping(linspace[0], -ones, linspace[1]),   # face #2, y-
            3: self.ct.mapping(linspace[0], ones, linspace[1]),    # face #3, y+
            4: self.ct.mapping(linspace[0], linspace[1], -ones),   # face #4, z-
            5: self.ct.mapping(linspace[0], linspace[1], ones),    # face #5, z+
        }

    def _generate_element_vtk_data_(self, xi, et, sg):
        """"""
        assert xi.ndim == et.ndim == sg.ndim == 1
        sx, sy, sz = xi.size, et.size, sg.size
        meshgrid = np.meshgrid(xi, et, sg, indexing='ij')
        X, Y, Z = self.ct.mapping(*meshgrid)
        coo_dict = {}
        for k in range(sz):
            for j in range(sy):
                for i in range(sx):
                    x, y, z = X[i, j, k], Y[i, j, k], Z[i, j, k]
                    key = f"%.7f-%.7f-%.7f" % (x, y, z)
                    coo_dict[key] = (x, y, z)
        cell_list = list()
        for k in range(sz - 1):
            for j in range(sy - 1):
                for i in range(sx - 1):
                    cell_list.append((
                        [
                            f"%.7f-%.7f-%.7f" % (X[i, j, k], Y[i, j, k], Z[i, j, k]),
                            f"%.7f-%.7f-%.7f" % (X[i+1, j, k], Y[i+1, j, k], Z[i+1, j, k]),
                            f"%.7f-%.7f-%.7f" % (X[i+1, j+1, k], Y[i+1, j+1, k], Z[i+1, j+1, k]),
                            f"%.7f-%.7f-%.7f" % (X[i, j+1, k], Y[i, j+1, k], Z[i, j+1, k]),
                            f"%.7f-%.7f-%.7f" % (X[i, j, k+1], Y[i, j, k+1], Z[i, j, k+1]),
                            f"%.7f-%.7f-%.7f" % (X[i+1, j, k+1], Y[i+1, j, k+1], Z[i+1, j, k+1]),
                            f"%.7f-%.7f-%.7f" % (X[i+1, j+1, k+1], Y[i+1, j+1, k+1], Z[i+1, j+1, k+1]),
                            f"%.7f-%.7f-%.7f" % (X[i, j+1, k+1], Y[i, j+1, k+1], Z[i, j+1, k+1]),
                        ], 8, 12)  # for this element, VTK_HEXAHEDRON cell (No. 12)!
                    )
        return coo_dict, cell_list

    def _generate_vtk_data_for_form(self, indicator, element_cochain, degree, data_density):
        """"""
        linspace = np.linspace(-1, 1, data_density)
        if indicator == 'm3n3k0':  # must be Lambda
            dtype = '3d-scalar'
            rc = ___rc330_msepy_quadrilateral___(
                self, degree, element_cochain, linspace, linspace, linspace, ravel=False)
        elif indicator == 'm3n3k1':   # must be Lambda
            dtype = '3d-vector'
            rc = ___rc331_orthogonal_hexahedron___(
                self, degree, element_cochain, linspace, linspace, linspace, ravel=False)
        elif indicator == 'm3n3k2':   # must be Lambda
            dtype = '3d-vector'
            rc = ___rc332_msepy_quadrilateral___(
                self, degree, element_cochain, linspace, linspace, linspace, ravel=False)
        elif indicator == 'm3n3k3':  # must be Lambda
            dtype = '3d-scalar'
            rc = ___rc333_msepy_quadrilateral___(
                self, degree, element_cochain, linspace, linspace, linspace, ravel=False)
        else:
            raise NotImplementedError()

        data_dict = {}

        if dtype == '3d-scalar':
            X, Y, Z, V = rc
            for i in range(data_density):
                for j in range(data_density):
                    for k in range(data_density):
                        x = X[i, j, k]
                        y = Y[i, j, k]
                        z = Z[i, j, k]
                        v = V[i, j, k]
                        key = "%.7f-%.7f-%.7f" % (x, y, z)
                        data_dict[key] = (x, y, z, v)

        elif dtype == '3d-vector':
            X, Y, Z, U, V, W = rc
            for i in range(data_density):
                for j in range(data_density):
                    for k in range(data_density):
                        x = X[i, j, k]
                        y = Y[i, j, k]
                        z = Z[i, j, k]
                        u = U[i, j, k]
                        v = V[i, j, k]
                        w = W[i, j, k]
                        key = "%.7f-%.7f-%.7f" % (x, y, z)
                        data_dict[key] = (x, y, z, u, v, w)
        else:
            raise NotImplementedError()

        cell_list = list()
        for i in range(data_density - 1):
            for j in range(data_density - 1):
                for k in range(data_density - 1):
                    cell_list.append((
                        [
                            "%.7f-%.7f-%.7f" % (X[i, j, k], Y[i, j, k], Z[i, j, k]),
                            "%.7f-%.7f-%.7f" % (X[i+1, j, k], Y[i+1, j, k], Z[i+1, j, k]),
                            "%.7f-%.7f-%.7f" % (X[i, j+1, k], Y[i, j+1, k], Z[i, j+1, k]),
                            "%.7f-%.7f-%.7f" % (X[i+1, j+1, k], Y[i+1, j+1, k], Z[i+1, j+1, k]),
                            "%.7f-%.7f-%.7f" % (X[i, j, k+1], Y[i, j, k+1], Z[i, j, k+1]),
                            "%.7f-%.7f-%.7f" % (X[i+1, j, k+1], Y[i+1, j, k+1], Z[i+1, j, k+1]),
                            "%.7f-%.7f-%.7f" % (X[i, j+1, k+1], Y[i, j+1, k+1], Z[i, j+1, k+1]),
                            "%.7f-%.7f-%.7f" % (X[i+1, j+1, k+1], Y[i+1, j+1, k+1], Z[i+1, j+1, k+1]),
                        ], 8, 11)   # VTK_VOXEL cell
                    )

        return data_dict, cell_list, dtype


# ============ ELEMENT CT =====================================================================================

class MseHtt_GreatMesh_OrthogonalHexahedron_Element_CooTrans(Frozen):
    """No need to use the standard CT form."""
    def __init__(
            self, element,
            origin_x, origin_y, origin_z,
            delta_x, delta_y, delta_z,
    ):
        self._element = element
        self._origin_x = origin_x
        self._origin_y = origin_y
        self._origin_z = origin_z
        self._ratio_x = delta_x / 2
        self._ratio_y = delta_y / 2
        self._ratio_z = delta_z / 2
        self._ct_helper = ___ct_helper_parser___(self._ratio_x, self._ratio_y, self._ratio_z)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<CT of {self._element.__repr__()}>"

    def mapping(self, xi, et, sg):
        """"""
        r = (xi + 1) * self._ratio_x
        s = (et + 1) * self._ratio_y
        t = (sg + 1) * self._ratio_z
        x = self._origin_x + r
        y = self._origin_y + s
        z = self._origin_z + t
        return x, y, z

    def Jacobian_matrix(self, xi, et, sg):
        """"""
        return self._ct_helper.Jacobian_matrix(xi, et, sg)

    def Jacobian(self, xi, et, sg):
        """"""
        return self._ct_helper.Jacobian(xi, et, sg)

    def inverse_Jacobian_matrix(self, xi, et, sg):
        """"""
        return self._ct_helper.inverse_Jacobian_matrix(xi, et, sg)

    def metric(self, xi, et, sg):
        """"""
        return self._ct_helper.metric(xi, et, sg)

    def inverse_Jacobian(self, xi, et, sg):
        """"""
        return self._ct_helper.inverse_Jacobian(xi, et, sg)

    def metric_matrix(self, xi, et, sg):
        """"""
        return self._ct_helper.metric_matrix(xi, et, sg)

    def inverse_metric_matrix(self, xi, et, sg):
        """"""
        return self._ct_helper.inverse_metric_matrix(xi, et, sg)


_cache_ct_helper_pool_ = {}


def ___ct_helper_parser___(ratio_x, ratio_y, ratio_z):
    """"""
    ratio_x = round(ratio_x, 8)
    ratio_y = round(ratio_y, 8)
    ratio_z = round(ratio_z, 8)
    key = (ratio_x, ratio_y, ratio_z)
    if key in _cache_ct_helper_pool_:
        pass
    else:
        _cache_ct_helper_pool_[key] = ___Orthogonal_Hexahedron___(*key)
    return _cache_ct_helper_pool_[key]


# noinspection PyUnusedLocal
class ___Orthogonal_Hexahedron___(Frozen):
    """"""
    def __init__(self, ratio_x, ratio_y, ratio_z):

        self._Jacobian_constant = ratio_x * ratio_y * ratio_z
        self._metric_constant = self._Jacobian_constant ** 2

        reciprocalJacobian_constant = 1 / self._Jacobian_constant
        inverse_Jacobian_matrix_00 = reciprocalJacobian_constant * ratio_y * ratio_z
        inverse_Jacobian_matrix_11 = reciprocalJacobian_constant * ratio_z * ratio_x
        inverse_Jacobian_matrix_22 = reciprocalJacobian_constant * ratio_x * ratio_y

        self._inverse_Jacobian_constant = (
                inverse_Jacobian_matrix_00 * inverse_Jacobian_matrix_11 * inverse_Jacobian_matrix_22
        )

        self._Jacobian_matrix = (
            (ratio_x, 0, 0),
            (0, ratio_y, 0),
            (0, 0, ratio_z),
        )
        self._iJM = (
            (inverse_Jacobian_matrix_00, 0, 0),
            (0, inverse_Jacobian_matrix_11, 0),
            (0, 0, inverse_Jacobian_matrix_22)
        )

        self._mm = (
            [ratio_x ** 2, 0, 0],
            [0, ratio_y ** 2, 0],
            [0, 0, ratio_z ** 2]
        )

        self._imm = (
            [inverse_Jacobian_matrix_00 ** 2, 0, 0],
            [0, inverse_Jacobian_matrix_11 ** 2, 0],
            [0, 0, inverse_Jacobian_matrix_22 ** 2]
        )

        self._freeze()

    def Jacobian_matrix(self, xi, et, sg):
        """"""
        return self._Jacobian_matrix

    def Jacobian(self, xi, et, sg):
        """"""
        return self._Jacobian_constant

    def inverse_Jacobian_matrix(self, xi, et, sg):
        """"""
        return self._iJM

    def metric(self, xi, et, sg):
        """"""
        return self._metric_constant

    def inverse_Jacobian(self, xi, et, sg):
        """"""
        return self._inverse_Jacobian_constant

    def metric_matrix(self, xi, et, sg):
        """"""
        return self._mm

    def inverse_metric_matrix(self, xi, et, sg):
        """"""
        return self._imm


# ============ FACES =====================================================================================

class MseHtt_GreatMesh_OrthogonalHexahedron_Element_Faces(Frozen):
    """"""
    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        """0, 1, 2, 3.

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
        assert face_id in range(6), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace(Frozen):
    """"""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace_CT(self)
        self._area = None
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct

    @property
    def area(self):
        """The area of this element face."""
        if self._area is None:
            nodes = np.array([-1, 1])
            x, y, z = self._ct.mapping(nodes, nodes, nodes)
            x0, x1 = x
            y0, y1 = y
            z0, z1 = z
            if x0 == x1:
                self._area = (y1 - y0) * (z1 - z0)
            elif y0 == y1:
                self._area = (z1 - z0) * (x1 - x0)
            elif z0 == z1:
                self._area = (x1 - x0) * (y1 - y0)
            else:
                raise Exception()
        return self._area


from msehtt.static.mesh.great.elements.types.base import _FaceCoordinateTransformationBase


class MseHtt_GreatMesh_OrthogonalHexahedron_Element_OneFace_CT(_FaceCoordinateTransformationBase):
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

    def mapping(self, xi, et, sg):
        r""""""
        m, n = self._axis, self._start_end
        if m == 0:  # x-direction
            ones = np.ones_like(et)
            if n == 0:  # x-
                return self._element.ct.mapping(-ones, et, sg)
            elif n == 1:  # x+
                return self._element.ct.mapping(ones, et, sg)
            else:
                raise Exception()
        elif m == 1:  # y-direction
            ones = np.ones_like(sg)
            if n == 0:  # y-
                return self._element.ct.mapping(xi, -ones, sg)
            elif n == 1:  # y+
                return self._element.ct.mapping(xi, ones, sg)
            else:
                raise Exception()
        elif m == 2:  # z-direction
            ones = np.ones_like(xi)
            if n == 0:  # z-
                return self._element.ct.mapping(xi, et, -ones)
            elif n == 1:  # z+
                return self._element.ct.mapping(xi, et, ones)
            else:
                raise Exception()
        else:
            raise Exception()

    def Jacobian_matrix(self, xi, et, sg):
        r""""""
        m, n = self._axis, self._start_end

        if m == 0:  # x-direction
            ones = np.ones_like(et)
            if n == 0:  # x-
                J = self._element.ct.Jacobian_matrix(-ones, et, sg)
            elif n == 1:  # x+
                J = self._element.ct.Jacobian_matrix(ones, et, sg)
            else:
                raise Exception()

            return ((J[0][1], J[0][2]),
                    (J[1][1], J[1][2]),
                    (J[2][1], J[2][2]))

        elif m == 1:  # y-direction
            ones = np.ones_like(sg)
            if n == 0:  # y-
                J = self._element.ct.Jacobian_matrix(xi, -ones, sg)
            elif n == 1:  # y+
                J = self._element.ct.Jacobian_matrix(xi, ones, sg)
            else:
                raise Exception()

            return ((J[0][2], J[0][0]),
                    (J[1][2], J[1][0]),
                    (J[2][2], J[2][0]))

        elif m == 2:  # z-direction
            ones = np.ones_like(xi)
            if n == 0:  # z-
                J = self._element.ct.Jacobian_matrix(xi, et, -ones)
            elif n == 1:  # z+
                J = self._element.ct.Jacobian_matrix(xi, et, ones)
            else:
                raise Exception()

            return ((J[0][0], J[0][1]),
                    (J[1][0], J[1][1]),
                    (J[2][0], J[2][1]))

        else:
            raise Exception()

    def outward_unit_normal_vector(self, xi, et, sg):
        r"""The outward unit norm vector (vec{n})."""
        J = self.Jacobian_matrix(xi, et, sg)
        a = (J[0][0], J[1][0], J[2][0])
        b = (J[0][1], J[1][1], J[2][1])
        acb0 = a[1] * b[2] - a[2] * b[1]
        acb1 = a[2] * b[0] - a[0] * b[2]
        acb2 = a[0] * b[1] - a[1] * b[0]
        norm = np.sqrt(acb0**2 + acb1**2 + acb2**2)

        nx = acb0 / norm
        ny = acb1 / norm
        nz = acb2 / norm

        n = self._start_end

        if n == 0:  # x-, y-, z- face
            return -nx, -ny, -nz
        else:
            return nx, ny, nz

    def is_plane(self):
        r""""""
        return True
