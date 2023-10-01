# -*- coding: utf-8 -*-
r"""
"""
from tools.frozen import Frozen
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
from generic.py._2d_unstruct.mesh.boundary_section.main import BoundarySection


class MseHyPy2MeshFaces(Frozen):
    """"""

    def __init__(self, msepy_boundary_section, current_elements):
        """"""
        self._background = msepy_boundary_section
        assert self.background.__class__ is MsePyBoundarySectionMesh, \
            f"msehy-py2-faces must have a background of msepy-boundary-section."
        assert self.background.base is current_elements.background, \
            (f"The background msepy-boundary-section must be based on the background (a msepy-mesh) "
             f"of current dependent elements.")
        self._generation = current_elements.generation
        including_element_faces = self._collect_element_faces(msepy_boundary_section, current_elements)
        self._generic = BoundarySection(current_elements.generic, including_element_faces)

        self._freeze()

    @property
    def generation(self):
        return self._generation

    @property
    def background(self):
        """"""
        return self._background

    def __repr__(self):
        """repr"""
        return rf"<G[{self.generation}] msehy2-boundary-faces UPON {self.background}>"

    @staticmethod
    def _collect_element_faces(msepy_boundary_section, current_elements):
        """"""
        msepy_faces = msepy_boundary_section.faces
        local_faces = msepy_faces._collect_local_faces()
        indices = current_elements.indices_in_base_element

        boundary_index_edge = list()

        for face in local_faces:
            element, m, n = face._element, face._m, face._n
            if m == 0 and n == 0:
                t_edge_indicator = '=2'
                q_edge_index = 3
            elif m == 0 and n == 1:
                t_edge_indicator = '=0'
                q_edge_index = 1
            elif m == 1 and n == 0:
                t_edge_indicator = '=3'
                q_edge_index = 0
            elif m == 1 and n == 1:
                t_edge_indicator = '=1'
                q_edge_index = 2
            else:
                raise Exception()

            local_indices = indices[element]
            for index in local_indices:
                if isinstance(index, str):
                    if t_edge_indicator in index:
                        num_level = index.count('-')
                        triangle = current_elements.levels[num_level].triangles[index]
                        local_map = triangle.local_map
                        if None in local_map:
                            assert local_map.count(None) == 1, f"A triangle element can only have one edge on boundary."
                            j = local_map.index(None)
                            if j == 0:
                                edge_index = 1
                            elif j == 1:
                                edge_index = 0
                            elif j == 2:
                                edge_index = 2
                            else:
                                raise Exception()

                            boundary_index_edge.append(
                                (index, edge_index)
                            )

                        else:
                            pass

                    else:
                        pass

                else:

                    boundary_index_edge.append(
                        (index, q_edge_index)
                    )
        return boundary_index_edge

    @property
    def generic(self):
        return self._generic

    def visualize(self, *args, title=None, **kwargs):
        if title is None:
            title = rf"${self.background.abstract._sym_repr}$"
        else:
            pass
        return self.generic.visualize(*args, title=title, **kwargs)
