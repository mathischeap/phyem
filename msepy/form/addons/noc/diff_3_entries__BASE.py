# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from scipy.sparse import csr_matrix
from tools.frozen import Frozen
from msepy.tools.matrix.dynamic import MsePyDynamicLocalMatrix
from msepy.tools.vector.dynamic import MsePyDynamicLocalVector
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from msepy.tools.vector.static.local import MsePyStaticLocalVector
from msepy.tools.nonlinear_operator.dynamic import MsePyDynamicLocalNonlinearOperator


class Base3Entries(Frozen):
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
            _vector_data_caller = _OneDimVector(self, vector_form)
            return MsePyDynamicLocalVector(_vector_data_caller), _vector_data_caller._time_caller

        elif dimensions == 2:

            row_form, col_form = args

            return _2d_matrix_representation(self._ABC, self._3d_data, row_form, col_form)

        elif dimensions == 3:

            if (self._A is not self._B) and (self._B is not self._C):
                return MsePyDynamicLocalNonlinearOperator(
                    self._3d_data,
                    self._A, self._B, self._C,
                    direct_derivative_contribution=True,
                )
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()


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

    caller = _D2MatrixCallerRowCol(ABC, _3d_data, row_index, col_index)

    return MsePyDynamicLocalMatrix(caller)


class _D2MatrixCallerRowCol(Frozen):
    """"""
    def __init__(self, ABC, data, row, col):
        """"""
        self._ABC = ABC
        self._data = data
        self._row = row
        self._col = col
        given_index = None
        for i in range(3):
            if i in (row, col):
                pass
            else:
                assert given_index is None
                given_index = i
        assert given_index is not None
        self._given_index = given_index
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        gm_row = self._ABC[self._row].cochain.gathering_matrix
        gm_col = self._ABC[self._col].cochain.gathering_matrix
        given_form = self._ABC[self._given_index]
        given_form_cochain = given_form.cochain._callable_cochain(*args, **kwargs)
        array_cochain = given_form_cochain.data
        _3d_data = self._data
        _2d_matrix_caller = _MatrixCaller(
            self._given_index, array_cochain, _3d_data, given_form.mesh, self._row, self._col
        )
        return MsePyStaticLocalMatrix(_2d_matrix_caller, gm_row, gm_col, cache_key='unique')


class _OneDimVector(Frozen):
    def __init__(self, ABC, vector_form):
        """"""
        assert vector_form in ABC._ABC, f"vector_form must one of the ABC forms."
        given_forms = list()
        given_indices = list()
        test_index = None
        for index, form in zip('ijk', ABC._ABC):
            if form is not vector_form:
                given_forms.append(form)
                given_indices.append(index)
            else:
                assert test_index is None, f"there is only one test form."
                test_index = index
        self._gfs = given_forms
        self._gis = given_indices
        self._tf = vector_form
        self._ti = test_index
        self._3d_data = ABC._3d_data
        self._freeze()

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

        local_vectors = list()
        for e in range(len(_3d_data)):  # all local elements
            data = _3d_data[e]

            local_vectors.append(
                np.einsum(
                    operands,
                    data, c0[e], c1[e],
                    optimize='optimal',
                )
            )

        local_vectors = np.vstack(local_vectors)

        return MsePyStaticLocalVector(local_vectors, self._tf.cochain.gathering_matrix)

    def _time_caller(self, *args, **kwargs):
        """"""
        times = list()
        for f in self._gfs:
            times.append(
                f.cochain._ati_time_caller(*args, **kwargs)
            )
        return times


class _MatrixCaller(Frozen):
    """"""

    def __init__(self, given_index, cochain, _3d_data, mesh, row_index, col_index):
        """"""
        self._cochain = cochain
        self._3d_data = _3d_data
        self._mesh = mesh
        self._given_key = 'ijk'[given_index]
        self._row_key = 'ijk'[row_index]
        self._col_key = 'ijk'[col_index]
        assert {self._given_key, self._row_key, self._col_key} == {'i', 'j', 'k'}, f"indices wrong!"
        self._freeze()

    def __call__(self, e):
        """return the static 2d matrix for element #e in real time."""
        M = np.einsum(
            f'ijk, {self._given_key} -> {self._row_key}{self._col_key}',
            self._3d_data[e],
            self._cochain[e],
            optimize='optimal'
        )
        M = csr_matrix(M)
        return M
