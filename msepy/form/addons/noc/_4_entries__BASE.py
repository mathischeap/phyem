# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from scipy.sparse import csr_matrix

from phyem.tools.frozen import Frozen
from phyem.msepy.tools.matrix.dynamic import MsePyDynamicLocalMatrix
from phyem.msepy.tools.vector.dynamic import MsePyDynamicLocalVector
from phyem.msepy.tools.vector.static.local import MsePyStaticLocalVector
from phyem.msepy.tools.matrix.static.local import MsePyStaticLocalMatrix


class Base4Entries(Frozen):
    """"""
    def __init__(self):
        self._A = None
        self._B = None
        self._C = None
        self._D = None
        self._ABCD = None
        self._4d_data = None
        self._mesh = None
        self._e2c = None
        self._type = None
        self._cache_key = None
        self._freeze()

    def _make_4d_data(self):
        """"""
        raise NotImplementedError()

    def __call__(self, dimensions, *args, **kwargs):
        """"""
        if self._4d_data is None:
            self._make_4d_data()
        else:
            pass

        if dimensions == 1:
            vector_form = args[0]
            # we reduce the 3d data into 1d according to vector-form's basis functions.
            _vector_data_caller = _B4E_1DimVector(self, vector_form)
            return MsePyDynamicLocalVector(_vector_data_caller), _vector_data_caller._time_caller

        elif dimensions == 2:
            row_form, col_form = args
            _matrix_data_caller, time_caller = _2d_matrix_representation(
                self._ABCD, self._4d_data,
                row_form, col_form
            )
            return _matrix_data_caller, time_caller

        else:
            raise NotImplementedError()


class _B4E_1DimVector(Frozen):
    def __init__(self, ABCD, vector_form):
        """"""
        assert vector_form in ABCD._ABCD, f"vector_form must one of the ABC forms."
        given_forms = list()
        given_indices = list()
        test_index = None
        for index, form in zip('ijkm', ABCD._ABCD):
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
        self._4d_data = ABCD._4d_data
        self._freeze()

    def __call__(self, *args, **kwargs):
        """Must return a mspy static local vector"""
        cochains = dict()
        times = self._time_caller(*args, **kwargs)
        for f, i, time in zip(self._gfs, self._gis, times):
            cochains[i] = f[time].cochain.local

        _4d_data = self._4d_data
        operands = f'ijkm, {self._gis[0]}, {self._gis[1]}, {self._gis[2]} -> {self._ti}'

        c0 = cochains[self._gis[0]]
        c1 = cochains[self._gis[1]]
        c2 = cochains[self._gis[2]]

        local_vectors = list()
        for e in range(len(_4d_data)):  # all local elements
            data = _4d_data[e]

            local_vectors.append(
                np.einsum(
                    operands,
                    data, c0[e], c1[e], c2[e],
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


def _2d_matrix_representation(ABCD, _4d_data, row_form, col_form):
    """We return a dynamic 2D matrix; in each matrix (for a local element), we have a 2d matrix
    whose row indices represent the basis functions of ``row_form`` and whose col indices represent
    the basis functions of ``col_form``.

    Parameters
    ----------
    ABCD
    _4d_data
    row_form
    col_form

    Returns
    -------

    """
    row_index = -1   # the row-form is self._ABC[row_index]
    col_index = -1   # the col-form is self._ABC[col_index]

    for i, form in enumerate(ABCD):
        if form is row_form:
            assert row_index == -1
            row_index = i
        if form is col_form:
            assert col_index == -1
            col_index = i
    assert row_index in (0, 1, 2, 3) and col_index in (0, 1, 2, 3) and row_index != col_index

    caller = _B4E_2DimMatrix(ABCD, _4d_data, row_index, col_index)

    matrix_caller = MsePyDynamicLocalMatrix(caller)
    time_caller = caller._time_caller

    return matrix_caller, time_caller


class _B4E_2DimMatrix(Frozen):
    """"""
    def __init__(self, ABCD, data, row, col):
        """"""
        self._ABCD = ABCD
        self._data = data
        self._row = row
        self._col = col
        given_index = list()
        self._gfs = list()
        for i in range(4):
            if i in (row, col):
                pass
            else:
                given_index.append(i)
                self._gfs.append(ABCD[i])
        assert len(given_index) == 2
        self._given_index = given_index
        self._freeze()

    def __call__(self, *args, **kwargs):
        """"""
        gm_row = self._ABCD[self._row].cochain.gathering_matrix
        gm_col = self._ABCD[self._col].cochain.gathering_matrix
        given_form_0 = self._ABCD[self._given_index[0]]
        given_form_1 = self._ABCD[self._given_index[1]]
        given_form_cochain0 = given_form_0.cochain._callable_cochain(*args, **kwargs)
        given_form_cochain1 = given_form_1.cochain._callable_cochain(*args, **kwargs)
        array_cochain0 = given_form_cochain0.data
        array_cochain1 = given_form_cochain1.data
        _4d_data = self._data
        _2d_matrix_caller = _MatrixCaller(
            self._given_index,
            array_cochain0, array_cochain1,
            _4d_data,
            given_form_0.mesh,
            self._row, self._col
        )
        return MsePyStaticLocalMatrix(_2d_matrix_caller, gm_row, gm_col, cache_key='unique')

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

    def __init__(self, given_index, cochain0, cochain1, _4d_data, mesh, row_index, col_index):
        """"""
        self._cochain0 = cochain0
        self._cochain1 = cochain1
        self._4d_data = _4d_data
        self._mesh = mesh
        self._given_key0 = 'ijkm'[given_index[0]]
        self._given_key1 = 'ijkm'[given_index[1]]
        self._row_key = 'ijkm'[row_index]
        self._col_key = 'ijkm'[col_index]
        assert {self._given_key0, self._given_key1, self._row_key, self._col_key} == {'i', 'j', 'k', 'm'}, \
            f"indices wrong!"
        self._freeze()

    def __call__(self, e):
        """return the static 2d matrix for element #e in real time."""
        M = np.einsum(
            f'ijkm, {self._given_key0}, {self._given_key1} -> {self._row_key}{self._col_key}',
            self._4d_data[e],
            self._cochain0[e], self._cochain1[e],
            optimize='optimal'
        )
        M = csr_matrix(M)
        return M
