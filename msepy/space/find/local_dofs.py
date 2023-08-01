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
        self._cache = dict()
        self._freeze()

    def __call__(self, m, n, degree):
        """Find the local dofs on `m`-axis, `n`-side of the space of degree `degree`.

        So m = {0, 1, 2, ...}, n = {0, 1}.

        """
        key = (m, n, degree)

        if key in self._cache:
            pass
        else:

            space = self._space
            indicator = space.abstract.indicator
            _k = space.abstract.k
            _n = space.abstract.n  # manifold dimensions
            _m = space.abstract.m  # dimensions of the embedding space.
            _orientation = space.abstract.orientation

            if indicator == 'Lambda':  # scalar valued form spaces

                if _m == _n == 1 and _k == 0:
                    local_dofs = self._Lambda_m1_n1_k0(m, n, degree)

                elif _m == _n == 2 and _k == 1 and _orientation == 'outer':
                    local_dofs = self._Lambda_m2_n2_k1_outer(m, n, degree)

                elif _m == _n == 2 and _k == 1 and _orientation == 'inner':
                    local_dofs = self._Lambda_m2_n2_k1_inner(m, n, degree)

                elif _m == _n == 2 and _k == 0:
                    local_dofs = self._Lambda_m2_n2_k0(m, n, degree)

                elif _m == _n == 3 and _k == 0:
                    local_dofs = self._Lambda_m3_n3_k0(m, n, degree)

                elif _m == _n == 3 and _k == 1:
                    local_dofs = self._Lambda_m3_n3_k1(m, n, degree)

                elif _m == _n == 3 and _k == 2:
                    local_dofs = self._Lambda_m3_n3_k2(m, n, degree)

                elif _m == _n == _k:
                    raise Exception(f"top-form has no dofs on element face!")

                else:
                    raise NotImplementedError

            else:
                raise NotImplementedError()

            self._cache[key] = local_dofs

        return self._cache[key]

    def _Lambda_m1_n1_k0(self, m, n, degree):
        """"""
        local_numbering = self._space.local_numbering(degree)[0]

        if m == 0:

            if n == 0:
                face_local_numbering = [local_numbering[0], ]  # local_numbering[0] is an int, so we put it in a list
            elif n == 1:
                face_local_numbering = [local_numbering[-1], ]  # local_numbering[-1] is an int, so we put it in a list
            else:
                raise Exception

        else:
            raise Exception

        return face_local_numbering

    def _Lambda_m2_n2_k1_outer(self, m, n, degree):
        """"""
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

        return face_local_numbering

    def _Lambda_m2_n2_k1_inner(self, m, n, degree):
        """"""
        local_numbering_dx, local_numbering_dy = self._space.local_numbering(degree)

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

        return face_local_numbering

    def _Lambda_m2_n2_k0(self, m, n, degree):
        """"""
        local_numbering = self._space.local_numbering(degree)[0]

        if m == 0:

            if n == 0:
                face_local_numbering = local_numbering[0, :]
            elif n == 1:
                face_local_numbering = local_numbering[-1, :]
            else:
                raise Exception

        elif m == 1:

            if n == 0:
                face_local_numbering = local_numbering[:, 0]
            elif n == 1:
                face_local_numbering = local_numbering[:, -1]
            else:
                raise Exception
        else:
            raise Exception

        return face_local_numbering

    def _Lambda_m3_n3_k0(self, m, n, degree):
        """"""
        local_numbering = self._space.local_numbering(degree)[0]

        if m == 0:

            if n == 0:
                face_local_numbering = local_numbering[0, :, :]
            elif n == 1:
                face_local_numbering = local_numbering[-1, :, :]
            else:
                raise Exception

        elif m == 1:

            if n == 0:
                face_local_numbering = local_numbering[:, 0, :]
            elif n == 1:
                face_local_numbering = local_numbering[:, -1, :]
            else:
                raise Exception

        elif m == 2:

            if n == 0:
                face_local_numbering = local_numbering[:, :, 0]
            elif n == 1:
                face_local_numbering = local_numbering[:, :, -1]
            else:
                raise Exception
        else:
            raise Exception

        return face_local_numbering

    def _Lambda_m3_n3_k1(self, m, n, degree):
        """"""
        local_numbering_dx, local_numbering_dy, local_numbering_dz = self._space.local_numbering(degree)

        if m == 0:
            local_numbering = local_numbering_dx
            if n == 0:
                face_local_numbering = local_numbering[0, :, :]
            elif n == 1:
                face_local_numbering = local_numbering[-1, :, :]
            else:
                raise Exception

        elif m == 1:
            local_numbering = local_numbering_dy
            if n == 0:
                face_local_numbering = local_numbering[:, 0, :]
            elif n == 1:
                face_local_numbering = local_numbering[:, -1, :]
            else:
                raise Exception

        elif m == 2:
            local_numbering = local_numbering_dz
            if n == 0:
                face_local_numbering = local_numbering[:, :, 0]
            elif n == 1:
                face_local_numbering = local_numbering[:, :, -1]
            else:
                raise Exception
        else:
            raise Exception

        return face_local_numbering

    def _Lambda_m3_n3_k2(self, m, n, degree):
        """"""
        local_numbering_dydz, local_numbering_dzdx, local_numbering_dxdy = self._space.local_numbering(degree)

        if m == 0:
            local_numbering = local_numbering_dydz
            if n == 0:
                face_local_numbering = local_numbering[0, :, :]
            elif n == 1:
                face_local_numbering = local_numbering[-1, :, :]
            else:
                raise Exception

        elif m == 1:
            local_numbering = local_numbering_dzdx
            if n == 0:
                face_local_numbering = local_numbering[:, 0, :]
            elif n == 1:
                face_local_numbering = local_numbering[:, -1, :]
            else:
                raise Exception

        elif m == 2:
            local_numbering = local_numbering_dxdy
            if n == 0:
                face_local_numbering = local_numbering[:, :, 0]
            elif n == 1:
                face_local_numbering = local_numbering[:, :, -1]
            else:
                raise Exception
        else:
            raise Exception

        return face_local_numbering
