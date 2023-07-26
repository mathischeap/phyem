# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:23 PM on 5/25/2023
"""
import numpy as np
from tools.frozen import Frozen
from tools.quadrature import Quadrature


class MsePyBoundarySectionFaces(Frozen):
    """"""

    def __init__(self, bs):
        """"""
        self._bs = bs
        self._initialize_elements()
        self._faces = dict()
        self._freeze()

    def _initialize_elements(self):
        """initialize_elements"""
        base = self._bs.base
        region_map = self._bs.manifold.regions.map
        base_map = base.manifold.regions.map
        base_elements_numbering = base.elements._numbering

        _elements = list()
        _m = list()
        _n = list()
        for ri in region_map:
            for j, mp in enumerate(region_map[ri]):
                if mp == 1:  # this boundary section covers this region face
                    assert base_map[ri][j] is None, f"safety check!"

                    m = j // 2
                    n = j % 2

                    if n == 0:
                        N = 0
                    elif n == 1:
                        N = -1
                    else:
                        raise Exception

                    if base.n == 2:  # the base mesh is 2-dimensional, the boundary section is 1-.
                        if m == 0:
                            elements = base_elements_numbering[ri][N, :]
                        elif m == 1:
                            elements = base_elements_numbering[ri][:, N]
                        else:
                            raise NotImplementedError()

                    else:
                        raise NotImplementedError()

                    _1d_elements = elements.ravel('F')
                    _1d_ones = np.ones_like(_1d_elements)
                    _elements.extend(_1d_elements)
                    _m.extend(_1d_ones * m)
                    _n.extend(_1d_ones * n)

                else:   # this boundary section does not cover this region face, just skip
                    pass

        self._shape = len(_elements)
        self._elements_m_n = np.array(
            [
                _elements, _m, _n
            ]
        )

    def __len__(self):
        """len."""
        return self._shape

    def __contains__(self, item):
        """contains. The local elements are indexed 0, 1, 2, 3, 4, ..."""
        if isinstance(item, (int, float)) and (0 <= item < len(self)) and item % 1 == 0:
            return True

        else:
            return False

    def __iter__(self):
        """"""
        for i in range(self._shape):
            yield i

    def __getitem__(self, i):
        """`i`th local element (element face of the base mesh)."""
        if i in self._faces:
            face = self._faces[i]

        else:
            assert i in self, f"i={i} is not a valid face number, must be in range(0, {len(self)})."
            element, m, n = self._elements_m_n[:, i]
            face = _MsePyBoundarySectionFace(
                i, self._bs, element, m, n
            )
            self._faces[i] = face

        return face


class _MsePyBoundarySectionFace(Frozen):
    """MsePyBoundarySectionFace"""

    def __init__(self, i, bs, element, m, n):
        """on the `n`(0, 1) side along `m` (0, 1, ...) axis of the element `#element`."""
        self._i = i
        self._bs = bs
        self._element = element
        self._m = m
        self._n = n
        self._ct = None
        self._freeze()

    @property
    def ct(self):
        """"""
        if self._ct is None:
            self._ct = self._bs.base.elements[self._element].ct.face(self._m, self._n)
        return self._ct

    def __repr__(self):
        """"""
        side = '+' if self._n == 1 else '-'
        self_repr = rf"<Face #{self._i} @ {side}side of {self._m}-axis of mesh element {self._element} on "
        bs_repr = self._bs.__repr__()
        return self_repr + bs_repr + '>'

    def find_corresponding_local_dofs_of(self, rf):
        """# find the local dofs of root-form ``rf`` on this element face."""
        return rf._find_local_dofs_on(self._m, self._n)
    
    @property
    def metric_signature(self):
        """A signature indicating the metric of the face. If it is None, then it is a unique face.
        Otherwise, all faces with the same signatures are of the same metric.
        """
        element_metric_signature = self._bs.base.elements[self._element].metric_signature
        if element_metric_signature is None:
            return None  # the element is unique, thus this face is unique as well.
        else:
            m, n = self._m, self._n
            return element_metric_signature + f" | <{m}-{n}>"

    def is_orthogonal(self):
        """Whether this face is perpendicular to coo axis."""
        element_metric_signature = self._bs.base.elements[self._element].metric_signature
        if element_metric_signature is None:  # when this face is unique, it is not orthogonal of course.
            return False
        else:
            return element_metric_signature[:6] == 'Linear'

    @property
    def length(self):
        """"""
        if self._bs.n == 1:

            if self.is_orthogonal():
                nodes = np.array([-1, 1])
                x, y = self.ct.mapping(nodes)
                x0, x1 = x
                y0, y1 = y
                length = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

            else:
                nodes, weights = Quadrature(10).quad
                Jx, Jy = self.ct.Jacobian_matrix(nodes)
                length = np.sum(np.sqrt(Jx**2 + Jy**2) * weights)

            return length

        else:
            raise Exception(f"only 1d boundary section face has a length!")
