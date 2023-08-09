# -*- coding: utf-8 -*-
"""
pH-lib@RAM-EEMCS-UT
Yi Zhang
Created at 5:40 PM on 7/26/2023
"""
import numpy as np
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker
from tools.quadrature import Quadrature
from msepy.tools.matrix.dynamic import MsePyDynamicLocalMatrix
from msepy.tools.matrix.static.local import MsePyStaticLocalMatrix
from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from scipy.sparse import csr_array


class _AxBipC(Frozen):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(AxB, C)"""
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        cache_key = list()
        for msepy_form in (A, B, C):
            assert msepy_form.__class__ is MsePyRootForm, f"{msepy_form} is not a {MsePyRootForm}!"
            cache_key.append(
                msepy_form.__repr__() + '@degree:' + _degree_str_maker(msepy_form.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._cache_key = cache_key

        self._A = A
        self._B = B
        self._C = C
        self._ABC = (A, B, C)
        if all([_.space.abstract.__class__ is ScalarValuedFormSpace for _ in self._ABC]):
            self._type = 'scalar'
        else:
            raise NotImplementedError()

        self._quad = quad
        self._3d_data = None
        self._mesh_dimensions = A.mesh.n
        self._mesh = A.mesh
        self._e2c = A.mesh.elements._index_mapping._e2c
        self._freeze()

    def _make_3d_data(self):
        """"""
        if self._3d_data is not None:
            return
        else:
            pass

        if self._quad is None:
            degrees = list()
            for form in self._ABC:
                degrees.append(
                    form.space[form._degree].p
                )
            degrees = np.array(degrees)
            degrees = np.max(degrees, axis=0)
            degrees = [int(_ * 1.5) + 1 for _ in degrees]
            types = 'Gauss'
        elif isinstance(self._quad, int):
            degrees = [self._quad for _ in range(self._mesh_dimensions)]
            types = 'Gauss'
        else:
            raise NotImplementedError()

        quad_degrees, quad_types = degrees, types

        quad = Quadrature(quad_degrees, category=types)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rmA = self._A.reconstruction_matrix(*quad_nodes)
        rmB = self._B.reconstruction_matrix(*quad_nodes)
        rmC = self._C.reconstruction_matrix(*quad_nodes)

        if self._mesh.n == 2:
            xi, et = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            detJ = self._mesh.elements.ct.Jacobian(xi, et)
        elif self._mesh.n == 3:
            xi, et, sg = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            sg = sg.ravel('F')
            detJ = self._mesh.elements.ct.Jacobian(xi, et, sg)
        else:
            raise Exception()

        _data_cache = dict()
        _3d_data = dict()
        for e in self._mesh.elements:

            cache_index = self._e2c[e]

            if cache_index in _data_cache:
                pass

            else:
                if self._type == 'scalar':
                    # make the data --------------- for 2d meshes -----------------------------------
                    if self._mesh.n == 2:
                        if len(rmA[e]) == 1 and len(rmB[e]) == len(rmC[e]) == 2:
                            # A is a 0-form, B, C are 1-forms!
                            # so, A = [0 0 w]^T, B = [u, v, 0]^T, C = [a b 0]^T,
                            # cp_term = A X B = [-wv wu 0]^T, (cp_term, C) = -wva + wub
                            w = rmA[e][0]
                            u, v = rmB[e]
                            a, b = rmC[e]
                            dJi = detJ[e]
                            data = - np.einsum(
                                'li, lj, lk, l -> ijk', w, v, a, quad_weights * dJi, optimize='optimal'
                            ) + np.einsum(
                                'li, lj, lk, l -> ijk', w, u, b, quad_weights * dJi, optimize='optimal'
                            )
                        elif len(rmA[e]) == len(rmB[e]) == 2 and len(rmC[e]) == 1:
                            # A, B are 1-forms, C is a 0-form!
                            # so, A = [wx wy, 0]^T    B = [u v 0]^T   C= [0 0 c]^T
                            # A x B = [wy*0 - 0*v,   0*u - wx*0,   wx*v - wy*u]^T = [0 0 C0]^T
                            # (A x B) dot C = 0*0 + 0*0 + C0*c = wx*v*c - wy*u*c
                            wx, wy = rmA[e]
                            u, v = rmB[e]
                            c = rmC[e][0]
                            dJi = detJ[e]
                            data = - np.einsum(
                                'li, lj, lk, l -> ijk', wx, v, c, quad_weights * dJi, optimize='optimal'
                            ) + np.einsum(
                                'li, lj, lk, l -> ijk', wy, u, c, quad_weights * dJi, optimize='optimal'
                            )

                        else:
                            raise NotImplementedError()

                    # make the data --------------- for 3d meshes -----------------------------------
                    elif self._mesh.n == 3:
                        if len(rmA[e]) == len(rmB[e]) == len(rmC[e]) == 3:
                            # A, B, C are all vectors.
                            # A = [wx wy, wz]^T    B = [u v w]^T   C= [a b c]^T
                            # A x B = [wy*w - wz*v,   wz*u - wx*w,   wx*v - wy*u]^T = [A0 B0 C0]^T
                            # (A x B) dot C = A0*a + B0*b + C0*c
                            wx, wy, wz = rmA[e]
                            u, v, w = rmB[e]
                            a, b, c = rmC[e]
                            dJi = detJ[e]
                            A0a = np.einsum(
                                'li, lj, lk, l -> ijk', wy, w, a, quad_weights * dJi, optimize='optimal'
                            ) - np.einsum(
                                'li, lj, lk, l -> ijk', wz, v, a, quad_weights * dJi, optimize='optimal'
                            )
                            B0b = np.einsum(
                                'li, lj, lk, l -> ijk', wz, u, b, quad_weights * dJi, optimize='optimal'
                            ) - np.einsum(
                                'li, lj, lk, l -> ijk', wx, w, b, quad_weights * dJi, optimize='optimal'
                            )
                            C0c = np.einsum(
                                'li, lj, lk, l -> ijk', wx, v, c, quad_weights * dJi, optimize='optimal'
                            ) - np.einsum(
                                'li, lj, lk, l -> ijk', wy, u, c, quad_weights * dJi, optimize='optimal'
                            )

                            data = A0a + B0b + C0c
                        else:
                            raise NotImplementedError()

                    # else: must be wrong, we do not do this in 1d ----------------------------------
                    else:
                        raise Exception()

                else:
                    raise NotImplementedError()

                _data_cache[cache_index] = data

            _3d_data[e] = _data_cache[cache_index]

        self._3d_data = _3d_data

    def __call__(self, dimensions, *args, **kwargs):
        """"""
        if self._3d_data is None:
            self._make_3d_data()
        else:
            pass

        if dimensions == 2:

            row_form, col_form = args

            return self._2d_matrix_representation(row_form, col_form)

        else:
            raise NotImplementedError()

    def _2d_matrix_representation(self, row_form, col_form):
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

        for i, form in enumerate(self._ABC):
            if form is row_form:
                assert row_index == -1
                row_index = i
            if form is col_form:
                assert col_index == -1
                col_index = i

        if row_index == 2 and col_index == 1:
            caller = self._2d_matrix_caller_r2_c1

        elif row_index == 2 and col_index == 0:
            caller = self._2d_matrix_caller_r2_c0

        else:
            raise NotImplementedError()

        return MsePyDynamicLocalMatrix(caller)

    def _2d_matrix_caller_r2_c1(self, *args, **kwargs):
        """This must return a `MsePyStaticLocalMatrix` object.

        As _3d_data does not change, ``*args, **kwargs`` will be used to determine the abstract time
        instant for the cochain of the given form. Then this cochain is used to make 2d data which are
        stored in a `MsePyStaticLocalMatrix`.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        gm_row = self._ABC[2].cochain.gathering_matrix
        gm_col = self._ABC[1].cochain.gathering_matrix
        given_form = self._ABC[0]
        given_form_cochain = given_form.cochain._callable_cochain(*args, **kwargs)
        array_cochain = given_form_cochain.data
        _3d_data = self._3d_data
        _2d_matrix_caller = _MatrixCaller(
            0, array_cochain, _3d_data, given_form.mesh, 2, 1
        )
        return MsePyStaticLocalMatrix(_2d_matrix_caller, gm_row, gm_col, cache_key='unique')

    def _2d_matrix_caller_r2_c0(self, *args, **kwargs):
        """This must return a `MsePyStaticLocalMatrix` object.

        As _3d_data does not change, ``*args, **kwargs`` will be used to determine the abstract time
        instant for the cochain of the given form. Then this cochain is used to make 2d data which are
        stored in a `MsePyStaticLocalMatrix`.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        gm_row = self._ABC[2].cochain.gathering_matrix
        gm_col = self._ABC[0].cochain.gathering_matrix
        given_form = self._ABC[1]
        given_form_cochain = given_form.cochain._callable_cochain(*args, **kwargs)
        array_cochain = given_form_cochain.data
        _3d_data = self._3d_data
        _2d_matrix_caller = _MatrixCaller(
            1, array_cochain, _3d_data, given_form.mesh, 2, 0
        )
        return MsePyStaticLocalMatrix(_2d_matrix_caller, gm_row, gm_col, cache_key='unique')


class _MatrixCaller(Frozen):
    """"""

    def __init__(self, given_index, cochain, _3d_data, mesh, row_index, col_index):
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
        M = csr_array(M)
        return M
