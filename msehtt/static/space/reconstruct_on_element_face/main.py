# -*- coding: utf-8 -*-
r"""Reconstruct on an element face/edge.
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.reconstruct_on_element_face.Lambda.main import MseHttSpace_RConEF_Lambda


class MseHttSpace_RConEF(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        indicator = self._space.indicator
        if indicator == 'Lambda':  # vector-valued form spaces.
            pass
            self._instance_for_space = MseHttSpace_RConEF_Lambda(space)
        else:
            raise NotImplementedError(f'ref (reconstruct on element face/edge) is not implemented '
                                      f'for space-type={indicator}')

        self._freeze()

    def __call__(self, degree, cochain, element_index, face_index, *mesh_grid):
        r"""Do the reconstruction along element face/edge."""
        return self._instance_for_space(degree, cochain, element_index, face_index, *mesh_grid)
