# -*- coding: utf-8 -*-
r"""
"""
import numpy as np

from phyem.tools.frozen import Frozen
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans

from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k2 import ___rc222_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221i_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k1 import ___rc221o_msepy_quadrilateral___
from phyem.msehtt.static.space.reconstruct.Lambda.Rc_m2n2k0 import ___rc220_msepy_quadrilateral___


class MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElement(MseHttGreatMeshBaseElement):
    """
    The real element is mapped from the following reference element.

   _________________________________> eta
    |  0        face #0       2
    |  -----------------------
    |  |                     |
    |  |         (ref)       |
    |  | face #2             |face #3
    |  |                     |
    |  -----------------------
    v  1      face #1        3
    xi

    The labels in _map refers to the for nodes in such a sequence.

    For example, _map = [87, 44, 156, 7561], then it is
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

    def __init__(self, element_index, parameters, _map, msepy_manifold):
        """"""
        self._region = parameters['region']
        self._origin = parameters['origin']
        self._delta = parameters['delta']
        assert msepy_manifold is not None, \
            f"unique msepy curvilinear quadrilateral must have the original msepy manifold."
        self._msepy_manifold = msepy_manifold
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementCooTrans(self, self.metric_signature)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<Unique Msepy Curvilinear quadrilateral element indexed:{self._index}" + super_repr

    @classmethod
    def m(cls):
        """the dimensions of the space"""
        return 2

    @classmethod
    def n(cls):
        """the dimensions of the element"""
        return 2

    @classmethod
    def _etype(cls):
        return 'unique msepy curvilinear quadrilateral'

    @property
    def metric_signature(self):
        """return int when it is unique."""
        return id(self)

    def _generate_outline_data(self, ddf=1):
        """"""
        if ddf <= 0.1:
            ddf = 0.1
        else:
            pass
        samples = 30 * ddf
        if samples >= 100:
            samples = 100
        elif samples < 5:
            samples = 5
        else:
            samples = int(samples)

        linspace = np.linspace(-1, 1, samples)
        ones = np.ones_like(linspace)

        return {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: self.ct.mapping(-ones, linspace),   # face #0
            1: self.ct.mapping(ones, linspace),    # face #1
            2: self.ct.mapping(linspace, -ones),   # face #2
            3: self.ct.mapping(linspace, ones),    # face #3
        }

    @classmethod
    def face_setting(cls):
        """To show the nodes of faces and the positive direction."""
        return {
            0: (0, 2),   # face #0 is from node 0 -> node 2  (positive direction)
            1: (1, 3),   # face #1 is from node 1 -> node 3  (positive direction)
            2: (0, 1),   # face #2 is from node 0 -> node 1  (positive direction)
            3: (2, 3),   # face #3 is from node 2 -> node 3  (positive direction)
        }

    @property
    def faces(self):
        if self._faces is None:
            self._faces = MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFaces(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([-1, 1, 0, 0])
        y = np.array([0, 0, -1, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (round(x[0], 7), round(y[0], 7)),
            1: r"%.7f-%.7f" % (round(x[1], 7), round(y[1], 7)),
            2: r"%.7f-%.7f" % (round(x[2], 7), round(y[2], 7)),
            3: r"%.7f-%.7f" % (round(x[3], 7), round(y[3], 7)),
        }

    @property
    def edges(self):
        raise Exception(f"msepy curvilinear quadrilateral element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"msepy curvilinear quadrilateral element has no edges.")

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

    def _generate_vtk_data_for_form(self, indicator, element_cochain, degree, data_density):
        """"""
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
                    key = "%.7f-%.7f" % (round(x, 7), round(y, 7))
                    data_dict[key] = (x, y, v)

        elif dtype == '2d-vector':
            X, Y, U, V = rc
            for i in range(data_density):
                for j in range(data_density):
                    x = X[i][j]
                    y = Y[i][j]
                    u = U[i][j]
                    v = V[i][j]
                    key = "%.7f-%.7f" % (round(x, 7), round(y, 7))
                    data_dict[key] = (x, y, u, v)
        else:
            raise NotImplementedError()

        cell_list = list()
        for i in range(data_density - 1):
            for j in range(data_density - 1):
                cell_list.append((
                    [
                        "%.7f-%.7f" % (round(X[i][j], 7), round(Y[i][j], 7)),
                        "%.7f-%.7f" % (round(X[i + 1][j], 7), round(Y[i + 1][j], 7)),
                        "%.7f-%.7f" % (round(X[i + 1][j + 1], 7), round(Y[i + 1][j + 1], 7)),
                        "%.7f-%.7f" % (round(X[i][j + 1], 7), round(Y[i][j + 1], 7)),
                    ], 4, 9)
                )

        return data_dict, cell_list, dtype


# ============ ELEMENT CT =====================================================================================
class MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementCooTrans(MseHttGreatMeshBaseElementCooTrans):
    """"""

    def mapping(self, xi, et):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate([xi, et]):
            _ = (_ + 1) * 0.5 * self._element._delta[j] + self._element._origin[j]
            md_ref_coo.append(_)

        return self._element._msepy_manifold.ct.mapping(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

    def ___Jacobian_matrix___(self, xi, et):
        """"""
        md_ref_coo = list()
        for j, _ in enumerate((xi, et)):
            _ = (_ + 1) * 0.5 * self._element._delta[j] + self._element._origin[j]
            md_ref_coo.append(_)

        jm = self._element._msepy_manifold.ct.Jacobian_matrix(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

        JM = tuple([
            [0, 0],
            [0, 0],
        ])
        for i in range(2):
            for j in range(2):
                jm_ij = jm[i][j]
                jm_ij *= self._element._delta[j] * 0.5
                JM[i][j] = jm_ij
        return JM


# ============ FACES ============================================================================================
class MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFaces(Frozen):
    """"""
    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        """0, 1, 2, 3.

       _________________________________> eta
        |  0        face #0       2
        |  -----------------------
        |  |                     |
        |  |         (ref)       |
        |  | face #2             |face #3
        |  |                     |
        |  -----------------------
        v  1      face #1        3
        xi

        """
        assert face_id in range(4), f"face id must be in range(4)."
        if face_id not in self._faces:
            self._faces[face_id] = MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFace(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFace(Frozen):
    """"""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFaceCT(self)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct


from phyem.msehtt.static.mesh.great.elements.types.orthogonal_rectangle import (
    MseHttGreatMeshOrthogonalRectangleElementFaceCT)


class MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElementFaceCT(
    MseHttGreatMeshOrthogonalRectangleElementFaceCT
):
    r""""""
    def __init__(self, face):
        r""""""
        super().__init__(face)
        self._melt()
        self.___is_place___ = None
        self._freeze()

    def is_plane(self):
        r""""""
        if self.___is_place___ is None:
            xi = np.linspace(-1, 1, 23)
            ounv = self.outward_unit_normal_vector(xi)
            n0, n1 = ounv
            if np.allclose(n0, n0[0]) and np.allclose(n1, n1[0]):
                self.___is_place___ = True
            else:
                self.___is_place___ = False
        return self.___is_place___
