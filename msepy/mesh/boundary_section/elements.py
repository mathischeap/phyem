# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:23 PM on 5/25/2023
"""
import numpy as np
from tools.frozen import Frozen


class MsePyBoundarySectionElements(Frozen):
    """"""

    def __init__(self, mesh):
        """"""
        self._mesh = mesh
        self._initialize_elements()
        self._faces = dict()
        self._freeze()

    def _initialize_elements(self):
        """initialize_elements"""
        base = self._mesh.base
        region_map = self._mesh.manifold.regions.map
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
                        pass
                    elif n == 1:
                        n = -1
                    else:
                        raise Exception

                    if self._mesh.n == 2:
                        if m == 0:
                            elements = base_elements_numbering[ri][n, :]
                        elif m == 1:
                            elements = base_elements_numbering[ri][:, n]
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

        self._elements = np.array(
            [
                _elements, _m, _n
            ]
        )

    def __len__(self):
        """len."""
        return self._shape

    def __contains__(self, item):
        """contains."""
        if isinstance(item, (int, float)) and (0 <= item < len(self)) and item % 1 == 0:
            return True

        else:
            return False

    def __getitem__(self, i):
        """`i`th element face."""
        if i in self._faces:
            face = self._faces[i]

        else:
            assert i in self, f"i={i} is not a valid face number, must be in range(0, {len(self)})."
            element, m, n = self._elements[:, i]
            face = _MsePyBoundarySectionFace(
                element, m, n
            )
            self._faces[i] = face

        return face


class _MsePyBoundarySectionFace(Frozen):
    """MsePyBoundarySectionFace"""

    def __init__(self, element, m, n):
        """on the `n`(0, 1) side along `m` (0, 1, ...) axis of the element `element`."""
        self._element = element
        self._m = m
        self._n = n
        self._freeze()
