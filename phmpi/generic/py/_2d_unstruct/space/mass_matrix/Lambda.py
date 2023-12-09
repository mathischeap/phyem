# -*- coding: utf-8 -*-
r"""
"""
from src.spaces.main import _degree_str_maker
from phmpi.generic.py.matrix.localize.static import MPI_PY_Localize_Static_Matrix
from legacy.generic.py._2d_unstruct.space.mass_matrix.Lambda import MassMatrixLambda


class MPI_PY_MassMatrixLambda(MassMatrixLambda):
    """"""

    def __call__(self, degree):
        """Making the local numbering for degree."""
        key = _degree_str_maker(degree)
        if key in self._cache:
            M = self._cache[key]
        else:
            k = self._k
            if k == 1:  # for k == 0 and k == 1.
                method_name = f"_k{k}_{self._orientation}"
            else:
                method_name = f"_k{k}"
            M = getattr(self, method_name)(degree)
            self._cache[key] = M  # M is the metadata, will never been touched.
        gm = self._space.gathering_matrix(degree)
        M = MPI_PY_Localize_Static_Matrix(M, gm, gm)
        return M
