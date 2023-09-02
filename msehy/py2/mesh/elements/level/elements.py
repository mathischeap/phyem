# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msehy.py2.mesh.elements.level.coordinate_tranformation import MseHyPy2MeshElementsLevelElementsCT
from msehy.py2.mesh.elements.level.triangle import MseHyPy2MeshElementsLevelElementsTriangle


class MseHyPy2MeshElementsLevelElements(Frozen):
    """"""

    def __init__(self, level):
        """"""
        self._level = level
        self._mesh = level._mesh
        self._level_num = level._level_num
        self._make_elements()
        self._ct = MseHyPy2MeshElementsLevelElementsCT(self)
        self._freeze()

    def __repr__(self):
        """repr"""
        return rf"<G[{self._level._mesh.___generation___}] elements of levels[{self._level_num}] of {self._mesh}>"

    @property
    def ct(self):
        """"""
        return self._ct

    def _make_elements(self):
        """"""
        indices = list()
        if self._level_num == 0:
            # ...
            _refining_elements = self._level._refining_elements

            for e in _refining_elements:
                indices.extend(
                    [f"{e}=0", f"{e}=1", f"{e}=2", f"{e}=3"]
                )

        else:
            raise NotImplementedError(f"level {self._level_num}.")

        self._triangle_dict = {}
        for ei in indices:
            self._triangle_dict[ei] = None

    def __getitem__(self, index):
        """Return the local element indexed `index` on this level."""
        assert index in self._triangle_dict, \
            f"element indexed {index} is not a valid element on level[{self._level_num}]."
        if self._triangle_dict[index] is None:
            # noinspection PyTypeChecker
            self._triangle_dict[index] = MseHyPy2MeshElementsLevelElementsTriangle(self, index)
        else:
            pass
        return self._triangle_dict[index]

    def __contains__(self, index):
        """if triangle ``index`` is a valid local triangle."""
        return index in self._triangle_dict

    def __iter__(self):
        """iter"""
        for index in self._triangle_dict:
            yield index

    def __len__(self):
        return len(self._triangle_dict)
