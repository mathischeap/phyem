# -*- coding: utf-8 -*-
r"""Reconstruct matrix along element face/edge.
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.reconstruction_matrix_for_element_face.Lambda.main import MseHttSpace_RM_eF_Lambda


class MseHttSpace_RMef(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        indicator = self._space.indicator
        if indicator == 'Lambda':  # vector-valued form spaces.
            self._instance_for_space = MseHttSpace_RM_eF_Lambda(space)
        else:
            raise NotImplementedError(f'ref (reconstruction along element face/edge) is not implemented '
                                      f'for space-type={indicator}')

        self._freeze()

    def __call__(self, degree, element_index, face_index, *mesh_grid):
        r"""Do the reconstruction along element face/edge."""
        return self._instance_for_space(degree, element_index, face_index, *mesh_grid)
