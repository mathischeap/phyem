# -*- coding: utf-8 -*-
r"""
"""
from phyem.tools.frozen import Frozen
from phyem.msehtt.static.space.local_dofs.Lambda.main import MseHttSpace_Local_Dofs_Lambda


class MseHttSpace_Local_Dofs(Frozen):
    r""""""

    def __init__(self, space):
        r""""""
        self._space = space
        self._freeze()

    def __call__(self, degree):
        r"""Return a dict, for example `D` whose
            keys:
                local element indices
            values:
                dictionaries

            So, for each key (each element), we have a dictionary. And this dictionary's keys are
            all local dof indices and values are the coordinate info of the local dof.
            For example,
            D[100] = {   # the local dof information of local element indexed 100.
                0: [local_coo_info, global_coo_info],
                1: ...,
                ...
            }
            So in the local element indexed 100, we have some local dofs locally labeled 0, 1, ...; and
            for the local dof #0, its coo_info in the reference element is `local_coo_info`, and its
            coo info in the physics domain is `global_coo_info`.

            If this dof is for a 0-form in m2n2, then it is like
            `local_coo_info = (-1, -1)` and `global_coo_info=(0, 0)`.
            It means this local dof is the top-left corner of the reference domain, and in the physical domain,
            it is at place (0, 0).

            Basically,

            If `local_coo_info=[float, float]` and `global_coo_info=[float, float]`,
            then we are looking at a nodal-dof in 2d space.

            And if `local_coo_info=[1d array, 1d-array]` and `global_coo_info=[1d array, 1d-array]`,
            then we are looking at an edge-dof in 2d space.

            And if `local_coo_info=[2d array, 2d-array]` and `global_coo_info=[2d array, 2d-array]`,
            then we are looking at a face-dof in 2d space.

            And if `local_coo_info=[float, float, float]` and `global_coo_info=[float, float, float]`,
            then we are looking at a nodal-dof in 3d space.

            And if `local_coo_info=[1d array, 1d-array, 1d-array]` and `global_coo_info=[1d array, 1d-array, 1d-array]`,
            then we are looking at an edge-dof in 3d space.

            And if `local_coo_info=[2d array, 2d-array, 2d-array]` and `global_coo_info=[2d array, 2d-array, 2d-array]`,
            then we are looking at a face-dof in 3d space.

            And if `local_coo_info=[3d array, 3d-array, 3d-array]` and `global_coo_info=[3d array, 3d-array, 3d-array]`,
            then we are looking at a volume-dof in 3d space.

        """
        indicator = self._space.indicator
        if indicator == 'Lambda':
            LG = MseHttSpace_Local_Dofs_Lambda(self._space)(degree)
        else:
            raise NotImplementedError()
        return LG
