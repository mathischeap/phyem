# -*- coding: utf-8 -*-
r"""
"""
from typing import Dict
from tools.frozen import Frozen
from msepy.mesh.boundary_section.main import MsePyBoundarySectionMesh
from msehy.py2.mesh.faces.visualize import MseHyPy2MeshFacesVisualize


class MseHyPy2MeshFaces(Frozen):
    """"""

    def __init__(self, msepy_mesh, current_elements):
        """"""
        self._background = msepy_mesh
        assert self.background.__class__ is MsePyBoundarySectionMesh, \
            f"msehy-py2-faces must have a background of msepy-boundary-section."
        assert self.background.base is current_elements.background, \
            (f"The background msepy-boundary-section must be based on the background (a msepy-mesh) "
             f"of current dependent elements.")
        self._elements = current_elements
        self._collecting_fundamental_faces()
        self._visualize = None
        self._freeze()

    @property
    def generation(self):
        return self._elements.generation

    @property
    def background(self):
        """"""
        return self._background

    @property
    def elements(self):
        return self._elements

    def __repr__(self):
        """repr"""
        return rf"<{self.generation}th generation msehy2-boundary-faces UPON {self.background.base}>"

    def _collecting_fundamental_faces(self):
        """"""
        base_ele_m_n = self.background.faces._elements_m_n.T
        base_ele_m_n = [tuple(_) for _ in base_ele_m_n]  # this can be cached and accelerated.

        self._fundamental_faces: Dict = dict()
        element_map = self.elements.map
        for i in element_map:
            _map = element_map[i]
            len_map = len(_map)
            if len_map == 3:  # this is a fundamental triangle cell
                assert isinstance(i, str)  # a fundamental triangle cell is indexed by string.
                for j, index in enumerate(['b', 0, 1]):
                    obj = _map[j]
                    if obj is None:
                        _split = i.split('=')
                        base_element = int(_split[0])
                        side_index = _split[1][0]
                        # we only need to check the first level to make sure which side this triangle is attached to
                        if side_index == '0':
                            m = 0
                            n = 1
                        elif side_index == '1':
                            m = 1
                            n = 1
                        elif side_index == '2':
                            m = 0
                            n = 0
                        elif side_index == '3':
                            m = 1
                            n = 0
                        else:
                            raise Exception()
                        location = (base_element, m, n)
                        if location in base_ele_m_n:
                            ff_index = (i, index)
                            self._fundamental_faces[ff_index] = None
                    else:
                        pass

            elif len_map == 4:  # this is a fundamental quadrilateral cell
                assert isinstance(i, int)  # a fundamental quadrilateral cell is indexed by integer.
                for j in range(4):
                    obj = _map[j]

                    if obj is None:
                        m = j // 2
                        n = j % 2
                        ff_index = (i, m, n)
                        if ff_index in base_ele_m_n:
                            self._fundamental_faces[ff_index] = None
                        else:
                            pass
                    else:
                        pass

            else:
                raise Exception()

    def __iter__(self):
        """go through all fundamental faces"""
        for index in self._fundamental_faces:
            yield index

    def __len__(self):
        """How many fundamental faces?"""
        return len(self._fundamental_faces)

    def __contains__(self, index):
        """If index indicating a fundamental face?"""
        return index in self._fundamental_faces

    def __getitem__(self, index):
        """Get the fundamental face instance of index ``index`."""
        if self._fundamental_faces[index] is None:
            self._fundamental_faces[index] = self.elements._get_boundary_fundamental_faces(index)
        return self._fundamental_faces[index]

    @property
    def visualize(self):
        """visualize"""
        if self._visualize is None:
            self._visualize = MseHyPy2MeshFacesVisualize(self)
        return self._visualize
