# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from legacy.generic.py.tools.nolinear_operation.base_3_entries import _OneDimVector
from legacy.generic.py.tools.nolinear_operation.base_3_entries import _Matrix_Caller
from legacy.generic.py.tools.nolinear_operation.base_3_entries import _D2_Matrix_Caller_Row_Col

from phmpi.generic.py.matrix.localize.dynamic import MPI_PY_Localize_Dynamic_Matrix
from phmpi.generic.py.matrix.localize.static import MPI_PY_Localize_Static_Matrix
from phmpi.generic.py.vector.localize.dynamic import MPI_PY_Localize_Dynamic_Vector
from phmpi.generic.py.vector.localize.static import MPI_PY_Localize_Static_Vector


class MPI_PY_Base3Entries(Frozen):
    """"""
    def __init__(self):
        self._3d_data = None
        self._A = None
        self._B = None
        self._C = None
        self._ABC = None

    def _make_3d_data(self):
        """"""
        raise NotImplementedError()

    def __call__(self, dimensions, *args, **kwargs):
        """"""
        if self._3d_data is None:
            self._make_3d_data()
        else:
            pass

        if dimensions == 1:
            vector_form = args[0]
            # we reduce the 3d data into 1d according to vector-form's basis functions.
            _vector_data_caller = _MPI_PY_OneDimVector(self, vector_form)
            return MPI_PY_Localize_Dynamic_Vector(_vector_data_caller), _vector_data_caller._time_caller

        elif dimensions == 2:
            row_form, col_form = args
            return _2d_matrix_representation(self._ABC, self._3d_data, row_form, col_form)

        elif dimensions == 3:
            # if (self._A is not self._B) and (self._B is not self._C):
            #     return MsePyDynamicLocalNonlinearOperator(
            #         self._3d_data,
            #         self._A, self._B, self._C,
            #         direct_derivative_contribution=True,
            #     )
            # else:
            raise NotImplementedError()

        else:
            raise NotImplementedError()


class _MPI_PY_OneDimVector(_OneDimVector):

    def __call__(self, *args, **kwargs):
        """Must return a mspy static local vector"""
        cochains = dict()
        times = self._time_caller(*args, **kwargs)
        for f, i, time in zip(self._gfs, self._gis, times):
            cochains[i] = f[time].cochain.local

        _3d_data = self._3d_data
        operands = f'ijk, {self._gis[0]}, {self._gis[1]} -> {self._ti}'

        c0 = cochains[self._gis[0]]
        c1 = cochains[self._gis[1]]

        local_vectors = dict()
        for index in _3d_data:  # all local elements
            data = _3d_data[index]

            local_vectors[index] = np.einsum(
                    operands,
                    data, c0[index], c1[index],
                    optimize='optimal',
                )

        return MPI_PY_Localize_Static_Vector(local_vectors, self._tf.cochain.gathering_matrix)


def _2d_matrix_representation(ABC, _3d_data, row_form, col_form):
    """We return a dynamic 2D matrix; in each matrix (for a local element), we have a 2d matrix
    whose row indices represent the basis functions of ``row_form`` and whose col indices represent
    the basis functions of ``col_form``.

    Parameters
    ----------
    row_form
    col_form

    Returns
    -------

    """
    row_index = -1   # the row-form is self._ABC[row_index]
    col_index = -1   # the col-form is self._ABC[col_index]

    for i, form in enumerate(ABC):
        if form is row_form:
            assert row_index == -1
            row_index = i
        if form is col_form:
            assert col_index == -1
            col_index = i
    assert row_index in (0, 1, 2) and col_index in (0, 1, 2) and row_index != col_index

    caller = MPI_PY_D2_Matrix_Caller_Row_Col(ABC, _3d_data, row_index, col_index)

    return MPI_PY_Localize_Dynamic_Matrix(caller)


class MPI_PY_D2_Matrix_Caller_Row_Col(_D2_Matrix_Caller_Row_Col):
    """"""
    def __call__(self, *args, **kwargs):
        """"""
        gm_row = self._ABC[self._row].cochain.gathering_matrix
        gm_col = self._ABC[self._col].cochain.gathering_matrix
        given_form = self._ABC[self._given_index]
        given_form_cochain_vector = given_form.cochain._dynamic_cochain_caller(*args, **kwargs)
        _3d_data = self._data
        _2d_matrix_caller = _Matrix_Caller(
            self._given_index, given_form_cochain_vector, _3d_data, given_form.mesh, self._row, self._col
        )
        return MPI_PY_Localize_Static_Matrix(_2d_matrix_caller, gm_row, gm_col, raw_key_map='unique')
