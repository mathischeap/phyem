# -*- coding: utf-8 -*-
r"""
"""
from phmpi.generic.py.matrix.localize.static import MPI_PY_Localize_Static_Matrix
from legacy.generic.py._2d_unstruct.space.incidence_matrix.Lambda import IncidenceMatrixLambda


class MPI_PY_IncidenceMatrixLambda(IncidenceMatrixLambda):
    """"""

    def __call__(self, degree):
        """"""
        p = self._space[degree].p
        if p in self._cache:
            E = self._cache[p]

        else:
            method_name = f"_k{self._k}_{self._orientation}"
            raw_E = getattr(self, method_name)(p)

            if self._k == 1:
                csm = self._space.basis_functions.csm(degree)
            elif self._k == 0:
                if self._orientation == 'inner':
                    csm = self._space.basis_functions._csm_Lambda_k1_inner(degree)
                elif self._orientation == 'outer':
                    csm = self._space.basis_functions._csm_Lambda_k1_outer(degree)
                else:
                    raise Exception
            else:
                raise Exception

            E = dict()
            for index in self._space.mesh:
                ele_type = self._space.mesh[index].type
                e = raw_E[ele_type]
                if index in csm:
                    if self._k == 1:
                        e = e @ csm[index]
                    elif self._k == 0:
                        e = csm[index] @ e
                    else:
                        raise Exception()
                else:
                    pass

                E[index] = e

            self._cache[p] = E

        gm_col = self._space.gathering_matrix(degree)
        indicator = self._space.abstract.indicator
        if indicator == 'Lambda':
            if self._k == 1:
                gm_row = self._space.gathering_matrix.Lambda._k2(degree)
            elif self._k == 0 and self._orientation == 'outer':
                gm_row = self._space.gathering_matrix.Lambda._k1_outer(degree)
            elif self._k == 0 and self._orientation == 'inner':
                gm_row = self._space.gathering_matrix.Lambda._k1_inner(degree)
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError()
        E = MPI_PY_Localize_Static_Matrix(E, gm_row, gm_col)
        return E
