# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
import numpy as np
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
from msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class MseHtt_GreatMesh_Unique_Msepy_Curvilinear_Triangle_Element(MseHttGreatMeshBaseElement):
    r"""
    First, we map the reference element into a reference triangle in the reference domain.

    So

   _________________________________> eta
    | (-1, -1)
    |  ----------------------- (-1, 1)
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  (1, -1)                 (1, 1)
    xi

    into

   _________________________________> eta
    | (-1, -1)
    |  ----------------------- (-1, 1)
    |  |                     |
    |  |     /\              |
    |  |    /  \             |
    |  |   /    \            |
    |  |  /______\           |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  (1, -1)                 (1, 1)
    xi

    then, into a curvilinear triangle in the physical domain. This transformation is same to the msepy
    unique transformation.

    As for the topology of the element, it is the same to that of vtu-5 triangle, i.e.,

    reference triangle: north edge into a point.

        edge west: edge 0
        edge east: edge 2
        edge south: edge 1
        node 0 (north edge node): (-1, 0)
        node 1 (west-south node): (1, -1)
        node 2 (east-south node): (1, 1)

    ______________________> et
    |           0        the north edge is collapsed into node 0
    |          /\
    |         /  \                 >   edge 0: positive direction: 0->1
    | edge0  /    \ edge 2         >>  edge 1: positive direction: 1->2
    |       /      \               >>> edge 2: positive direction: 0->2
    |      /        \
    |     ------------
    v     1   edge1   2
    xi

    """

    def __init__(self, element_index, parameters, _map, msepy_manifold):
        """"""
        self._region = parameters['region']
        self._origin = parameters['origin']
        self._delta = parameters['delta']
        self._xy = parameters['xy']
        assert msepy_manifold is not None, \
            f"unique msepy curvilinear triangle must have the original msepy manifold."
        self._msepy_manifold = msepy_manifold
        super().__init__()
        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = _Curvilinear_Triangle_CT_(self)

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
        return 'unique msepy curvilinear triangle'

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
            0: self.ct.mapping(linspace, -ones),   # face #0
            1: self.ct.mapping(ones, linspace),    # face #1
            2: self.ct.mapping(linspace, ones),    # face #2
        }

    @classmethod
    def face_setting(cls):
        """To show the nodes of faces and the positive direction."""
        return {
            0: (0, 1),   # face #0 is from node 0 -> node 1  (positive direction)
            1: (1, 2),   # face #1 is from node 1 -> node 2  (positive direction)
            2: (0, 2),   # face #2 is from node 0 -> node 2  (positive direction)
        }

    @property
    def faces(self):
        if self._faces is None:
            self._faces = _Curvilinear_Triangle_Faces_(self)
        return self._faces

    def ___face_representative_str___(self):
        r""""""
        x = np.array([0, 1, 0])
        y = np.array([-1, 0, 1])
        x, y = self.ct.mapping(x, y)
        return {
            0: r"%.7f-%.7f" % (x[0], y[0]),
            1: r"%.7f-%.7f" % (x[1], y[1]),
            2: r"%.7f-%.7f" % (x[2], y[2]),
        }

    @property
    def edges(self):
        raise Exception(f"msepy curvilinear triangle element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"msepy curvilinear triangle element has no edges.")


# ============ ELEMENT CT =====================================================================================


from msehtt.static.mesh.great.elements.types.vtu_5_triangle import ___invA___


class _Curvilinear_Triangle_CT_(MseHttGreatMeshBaseElementCooTrans):
    r"""The transformation of curvilinear triangle is like below:

    ________________________________> eta
    | (-1, -1)
    |  ----------------------- (-1, 1)
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  (1, -1)                 (1, 1)
    xi

    into

    ________________________________> eta
    |       (-1, 0)
    |         /\
    |        /  \
    |       /    \
    |      /      \
    |     /        \
    |    /          \
    |   /            \
    |  /              \
    | -----------------
    v  (1, -1)         (1, 1)
    xi

    into

    ________________________________> eta
    | (-1, -1)
    |  ----------------------- (-1, 1)
    |  |                     |
    |  |     /\              |
    |  |    /  \             |
    |  |   /    \            |
    |  |  /______\           |
    |  |                     |
    |  |                     |
    |  -----------------------
    v  (1, -1)                 (1, 1)
    xi

    Then we use the msepy unique mapping to map it into the curvilinear triangle in the physical domain.
    """

    def __init__(self, cte):
        """"""
        super().__init__(cte, cte.metric_signature)

        self._melt()

        # reference triangle: north edge into a point.
        # edge west: edge 0
        # edge east: edge 2
        # edge south: edge 1
        # node 0 (north edge node): (-1, 0)
        # node 1 (west-south node): (1, -1)
        # node 2 (east-south node): (1, 1)
        # A matrix =
        #   [ -1   1   1]
        #   [  0  -1   1]
        #   [  1   1   1]
        # inv(A) =
        #   [ -0.5,    0, 0.5 ]
        #   [ 0.25, -0.5, 0.25]
        #   [ 0.25,  0.5, 0.25]
        #

        X = np.vstack((np.array(
            cte._xy
        ), [1., 1., 1.]))

        T = X @ ___invA___

        self._t00 = T[0, 0]
        self._t01 = T[0, 1]
        self._t02 = T[0, 2]

        self._t10 = T[1, 0]
        self._t11 = T[1, 1]
        self._t12 = T[1, 2]

        self._freeze()

    def mapping(self, xi, et):
        """"""
        # [-1, 1]^2 -> a standard reference triangle.
        r = xi
        s = et * (xi + 1) / 2
        # --- then to the reference triangle.
        a = self._t00 * r + self._t01 * s + self._t02 * 1
        b = self._t10 * r + self._t11 * s + self._t12 * 1

        md_ref_coo = list()
        for j, _ in enumerate([a, b]):
            _ = (_ + 1) * 0.5 * self._element._delta[j] + self._element._origin[j]
            md_ref_coo.append(_)

        return self._element._msepy_manifold.ct.mapping(
            *md_ref_coo, regions=self._element._region
        )[self._element._region]

    def ___Jacobian_matrix___(self, xi, et):
        """"""
        # [-1, 1]^2 -> a standard reference triangle.
        r = xi
        s = et * (xi + 1) / 2
        # --- then to the reference triangle.
        a = self._t00 * r + self._t01 * s + self._t02 * 1
        b = self._t10 * r + self._t11 * s + self._t12 * 1

        dx_dr = self._t00
        dx_ds = self._t01

        dy_dr = self._t10
        dy_ds = self._t11

        dr_dxi = 1
        ds_dxi = et / 2
        ds_det = (xi + 1) / 2

        da_dxi = dx_dr * dr_dxi + dx_ds * ds_dxi
        da_det = dx_ds * ds_det

        db_dxi = dy_dr * dr_dxi + dy_ds * ds_dxi
        db_det = dy_ds * ds_det

        md_ref_coo = list()
        for j, _ in enumerate([a, b]):
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
                JM[i][j] = jm[i][j] * (self._element._delta[j] / 2)

        dx_da = JM[0][0]
        dx_db = JM[0][1]
        dy_da = JM[1][0]
        dy_db = JM[1][1]

        dx_dxi = dx_da * da_dxi + dx_db * db_dxi
        dx_det = dx_da * da_det + dx_db * db_det

        dy_dxi = dy_da * da_dxi + dy_db * db_dxi
        dy_det = dy_da * da_det + dy_db * db_det

        return (
            [dx_dxi, dx_det],
            [dy_dxi, dy_det],
        )


# ============ FACES ============================================================================================


class _Curvilinear_Triangle_Faces_(Frozen):
    """"""
    def __init__(self, element):
        """"""
        self._element = element
        self._faces = {}
        self._freeze()

    def __getitem__(self, face_id):
        r"""face_id in (0, 1, 2).

        ______________________> et
        |           0        the north edge is collapsed into node 0
        |          /\
        |         /  \                 >   edge 0: positive direction: 0->1
        | face0  /    \ face2          >>  edge 1: positive direction: 1->2
        |       /      \               >>> edge 2: positive direction: 0->2
        |      /        \
        |     ------------
        v     1   face1   2
        xi

        """
        assert face_id in range(3), f"face id must be in range(3)."
        if face_id not in self._faces:
            self._faces[face_id] = _Curvilinear_Triangle_OneFace_(self._element, face_id)
        else:
            pass
        return self._faces[face_id]

    def __repr__(self):
        """"""
        return f"<Faces of {self._element}>"


class _Curvilinear_Triangle_OneFace_(Frozen):
    r""""""
    def __init__(self, element, face_id):
        self._element = element
        self._id = face_id
        self._ct = _Curvilinear_Triangle_OneFace_CT_(self)
        self._freeze()

    def __repr__(self):
        """"""
        return f"<Face#{self._id} of {self._element}>"

    @property
    def ct(self):
        """Coordinate transformation of this face."""
        return self._ct


from msehtt.static.mesh.great.elements.types.vtu_5_triangle import _VTU5_Triangle_OneFace_CT_


class _Curvilinear_Triangle_OneFace_CT_(_VTU5_Triangle_OneFace_CT_):
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
