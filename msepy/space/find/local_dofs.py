# -*- coding: utf-8 -*-
r"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
"""
from tools.frozen import Frozen


class _MsePySpaceFindLocalDofs(Frozen):
    """"""

    def __init__(self, space):
        self._space = space
        self._cache_m2n2k1o = dict()
        self._freeze()

    def __call__(self, m, n, degree):
        """"""
        space = self._space
        indicator = space.abstract.indicator
        _k = space.abstract.k
        _n = space.abstract.n  # manifold dimensions
        _m = space.abstract.m  # dimensions of the embedding space.
        _orientation = space.abstract.orientation

        if indicator == 'Lambda':  # scalar valued form spaces

            if _m == _n == 2 and _k == 1 and _orientation == 'outer':

                return self._Lambda_m2_n2_k1_outer(m, n, degree)

            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()

    def _Lambda_m2_n2_k1_outer(self, m, n, degree):
        """"""
        cache_key = (m, n, degree)
        if cache_key in self._cache_m2n2k1o:
            pass
        else:

            local_numbering_dy, local_numbering_dx = self._space.local_numbering(degree)

            if m == 0:
                local_numbering = local_numbering_dy

                if n == 0:
                    face_local_numbering = local_numbering[0, :]
                elif n == 1:
                    face_local_numbering = local_numbering[-1, :]
                else:
                    raise Exception

            elif m == 1:
                local_numbering = local_numbering_dx

                if n == 0:
                    face_local_numbering = local_numbering[:, 0]
                elif n == 1:
                    face_local_numbering = local_numbering[:, -1]
                else:
                    raise Exception
            else:
                raise Exception

            self._cache_m2n2k1o[cache_key] = face_local_numbering

        return self._cache_m2n2k1o[cache_key]
