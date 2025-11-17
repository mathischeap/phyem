# -*- coding: utf-8 -*-
r"""Like the unique curvilinear quadrilateral element, but this element is a triangle element.
"""
import numpy as np

from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement


class Unique_Curvilinear_Triangle(MseHttGreatMeshBaseElement):
    r"""
    First, we map the reference element into a reference triangle in the reference domain.

    So

    ------------------------------------> eta
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
    |  (1, -1)                 (1, 1)
    v
    xi

    into (the north edge is collapsed into a node)

    --------------------------------------> eta
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
    |  (1, -1)                 (1, 1)
    v
    xi

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
    |     /__________\
    v     1   edge1   2
    xi

    """

    def __init__(self, element_index, parameters, _map):
        r""""""
        self._mp = parameters['mapping']           # refer quad into the triangle in physical domain
        self._JM = parameters['Jacobian_matrix']   # refer quad into the triangle in physical domain
        super().__init__()

        self._index = element_index
        self._parameters = parameters
        self._map = _map
        self._ct = UCT_CT(self)

    def __repr__(self):
        """"""
        super_repr = super().__repr__().split('object')[1]
        return f"<U-C-T element indexed:{self._index}" + super_repr

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
        return 'unique curvilinear triangle'

    @classmethod
    def _find_element_center_coo(cls, parameters):
        r""""""
        mp = parameters['mapping']
        x, y = mp(0, 0)
        return np.array([x, y])

    @classmethod
    def _find_mapping_(cls, parameters, x, y):
        r""""""
        return parameters['mapping'](x, y)

    @property
    def metric_signature(self):
        """return int when it is unique."""
        return id(self)

    def _generate_outline_data(self, ddf=1, internal_grid=0):
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

        line_dict = {
            'mn': (self.m(), self.n()),
            'center': self.ct.mapping(0, 0),
            0: self.ct.mapping(linspace, -ones),   # face #0
            1: self.ct.mapping(ones, linspace),    # face #1
            2: self.ct.mapping(linspace, ones),    # face #2
        }

        if internal_grid == 0:
            return line_dict
        else:
            raise NotImplementedError()

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
        """The faces of this element."""
        if self._faces is None:
            raise NotImplementedError()
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
        raise Exception(f"U-C-T (2d) element has no edges.")

    def ___edge_representative_str___(self):
        r""""""
        raise Exception(f"U-C-T (2d) element has no edges.")


from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElementCooTrans


class UCT_CT(MseHttGreatMeshBaseElementCooTrans):
    r""""""

    def __init__(self, ucq):
        """"""
        super().__init__(ucq, ucq.metric_signature)

    def mapping(self, xi, et):
        """"""
        return self._element._mp(xi, et)

    def ___Jacobian_matrix___(self, xi, et):
        """"""
        return self._element._JM(xi, et)
