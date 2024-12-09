# -*- coding: utf-8 -*-
r"""Reconstruct along element face/edge.
"""
from tools.frozen import Frozen
from msehtt.static.space.reconstruct_element_face.Lambda.main import MseHttSpace_REF_Lambda


class MseHttSpace_REF(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        indicator = self._space.indicator
        if indicator == 'Lambda':  # vector-valued form spaces.
            self._instance_for_space = MseHttSpace_REF_Lambda(space)
        else:
            raise NotImplementedError(f'ref (reconstruction along element face/edge) is not implemented '
                                      f'for space-type={indicator}')

        self._freeze()

    def __call__(self, degree, element_index, face_index, *mesh_grid):
        r"""Do the reconstruction along element face/edge."""
        return self._instance_for_space(degree, element_index, face_index, *mesh_grid)
