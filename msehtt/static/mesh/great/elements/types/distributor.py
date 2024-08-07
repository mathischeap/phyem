# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement
from msehtt.static.mesh.great.elements.types.unique_msepy_curvilinear_quadrilateral import (
    MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElement)

from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement


class MseHttGreatMeshElementDistributor(Frozen):
    """"""

    def __init__(self):
        """"""
        self._freeze()

    def __call__(
            self, element_index, etype, parameters, _map,
            msepy_manifold=None,
    ):
        """"""
        assert etype in self.implemented_element_types(), f"element type = {etype} is not implemented."
        element_class = self.implemented_element_types()[etype]
        if etype == 'unique msepy curvilinear quadrilateral':
            return element_class(element_index, parameters, _map, msepy_manifold)
        else:
            return element_class(element_index, parameters, _map)

    @classmethod
    def implemented_element_types(cls):
        """"""
        return {
            # m2n2 elements:
            'unique msepy curvilinear quadrilateral': MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElement,
            'orthogonal rectangle': MseHttGreatMeshOrthogonalRectangleElement,

            # m3n3 elements:
            'orthogonal hexahedron': MseHttGreatMeshOrthogonalHexahedronElement
        }
