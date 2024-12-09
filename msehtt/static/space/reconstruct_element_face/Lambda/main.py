# -*- coding: utf-8 -*-
r"""Reconstruction along element face/edge for Lambda-type spaces (scalar-valued form spaces)
"""
from tools.frozen import Frozen
from msehtt.static.space.reconstruct_element_face.Lambda.ref_m2n2k0 import ___ref_m2n2k0___
from msehtt.static.space.reconstruct_element_face.Lambda.ref_m2n2k1 import ___ref_m2n2k1_inner___
from msehtt.static.space.reconstruct_element_face.Lambda.ref_m2n2k1 import ___ref_m2n2k1_outer___
from msehtt.static.space.reconstruct_element_face.Lambda.ref_m2n2k2 import ___ref_m2n2k2___


class MseHttSpace_REF_Lambda(Frozen):
    """"""

    def __init__(self, space):
        """"""
        self._space = space
        m = self._space.m
        n = self._space.n
        k = self._space.abstract.k
        self._orientation = self._space.orientation
        self._indicator = f"m{m}n{n}k{k}"
        self._freeze()

    def __call__(self, degree, element_index, face_index, *mesh_grid):
        r""""""
        elements = self._space.tpm.composition
        element = elements[element_index]

        if self._indicator == 'm2n2k0':
            return ___ref_m2n2k0___(degree, element, face_index, *mesh_grid)
        elif self._indicator == 'm2n2k1':
            if self._orientation == 'inner':
                return ___ref_m2n2k1_inner___(degree, element, face_index, *mesh_grid)
            elif self._orientation == 'outer':
                return ___ref_m2n2k1_outer___(degree, element, face_index, *mesh_grid)
            else:
                raise Exception()
        elif self._indicator == 'm2n2k2':
            return ___ref_m2n2k2___(degree, element, face_index, *mesh_grid)
        else:
            raise NotImplementedError()
