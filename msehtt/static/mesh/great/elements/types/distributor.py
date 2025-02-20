# -*- coding: utf-8 -*-
r"""
"""

from tools.frozen import Frozen
from msehtt.static.mesh.great.elements.types.orthogonal_rectangle import MseHttGreatMeshOrthogonalRectangleElement
from msehtt.static.mesh.great.elements.types.unique_msepy_curvilinear_quadrilateral import (
    MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElement)
from msehtt.static.mesh.great.elements.types.unique_msepy_curvilinear_triangle import (
    MseHtt_GreatMesh_Unique_Msepy_Curvilinear_Triangle_Element)
from msehtt.static.mesh.great.elements.types.unique_curvilinear_quad import UniqueCurvilinearQuad
from msehtt.static.mesh.great.elements.types.unique_curvilinear_triangle import Unique_Curvilinear_Triangle

from msehtt.static.mesh.great.elements.types.orthogonal_hexahedron import MseHttGreatMeshOrthogonalHexahedronElement

from msehtt.static.mesh.great.elements.types.vtu_5_triangle import Vtu5Triangle
from msehtt.static.mesh.great.elements.types.vtu_8_pixel import Vtu8Pixel
from msehtt.static.mesh.great.elements.types.vtu_9_quad import Vtu9Quad
from msehtt.static.mesh.great.elements.types.vtu_11_voxel import Vtu_11_Voxel


class MseHttGreatMeshElementDistributor(Frozen):
    r""""""

    def __init__(self):
        r""""""
        self._freeze()

    def __call__(
            self, element_index, etype, parameters, _map,
            msepy_manifold=None,
    ):
        r""""""
        assert etype in self.implemented_element_types(), f"element type = {etype} is not implemented."
        element_class = self.implemented_element_types()[etype]
        if etype == 'unique msepy curvilinear quadrilateral':
            return element_class(element_index, parameters, _map, msepy_manifold)
        elif etype == 'unique msepy curvilinear triangle':
            return element_class(element_index, parameters, _map, msepy_manifold)
        else:
            return element_class(element_index, parameters, _map)

    @classmethod
    def implemented_element_types(cls):
        r""""""
        return {
            # m2n2 elements:
            'unique msepy curvilinear quadrilateral': MseHttGreatMeshUniqueMsepyCurvilinearQuadrilateralElement,
            'unique msepy curvilinear triangle': MseHtt_GreatMesh_Unique_Msepy_Curvilinear_Triangle_Element,
            'orthogonal rectangle': MseHttGreatMeshOrthogonalRectangleElement,
            'unique curvilinear quad': UniqueCurvilinearQuad,
            'unique curvilinear triangle': Unique_Curvilinear_Triangle,

            # m3n3 elements:
            'orthogonal hexahedron': MseHttGreatMeshOrthogonalHexahedronElement,
            11: Vtu_11_Voxel,   # same to 'orthogonal hexahedron'

            # 2d vtu elements:
            5: Vtu5Triangle,
            8: Vtu8Pixel,   # same to 'orthogonal rectangle'
            9: Vtu9Quad,

        }
