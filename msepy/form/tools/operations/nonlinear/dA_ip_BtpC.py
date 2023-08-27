# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from tools.frozen import Frozen
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker
from src.spaces.continuous.bundle import BundleValuedFormSpace
from tools.quadrature import Quadrature
from msepy.tools.matrix.dynamic import MsePyDynamicLocalMatrix
from msepy.form.tools.operations.nonlinear.AxB_ip_C import _D2MatrixCallerRow2Col0, _D2MatrixCallerRow2Col1


# noinspection PyPep8Naming
class __dA_ip_BtpC__(Frozen):
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
        if all([_.space.abstract.__class__ is BundleValuedFormSpace for _ in self._ABC]):
            self._type = 'bundle'
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

        if self._quad is None and self._type == 'bundle':
            n = self._mesh.n
            Degrees = list()
            for _ in range(n):
                degrees = list()
                for form in self._ABC:
                    degrees.append(
                        form.space[form._degree].p[_]
                    )
                degrees = np.array(degrees)
                degrees = np.max(degrees, axis=0)
                degrees = [int(_ * 1.5) + 1 for _ in degrees]

                Degrees.append(degrees)
            Degrees = np.array(Degrees)
            degrees = np.max(Degrees, axis=0)
            types = 'Gauss'

        else:
            raise NotImplementedError()

        quad_degrees, quad_types = degrees, types

        quad = Quadrature(quad_degrees, category=types)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rm_B = self._B.reconstruction_matrix(*quad_nodes)
        rm_C = self._C.reconstruction_matrix(*quad_nodes)

        EA = self._A.coboundary.incidence_matrix._data.toarray()
        dA = self._A.coboundary._make_df()
        rm_dA = dA.reconstruction_matrix(*quad_nodes)

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
                if self._type == 'bundle':
                    # make the data --------------- for 2d meshes -----------------------------------
                    if self._mesh.n == 2:

                        dJi = detJ[e]

                        A0, A1 = rm_dA[e]
                        A00, A01 = A0
                        A10, A11 = A1

                        dA00 = A00 @ EA
                        dA01 = A01 @ EA
                        dA10 = A10 @ EA
                        dA11 = A11 @ EA

                        b0, b1 = rm_B[e]
                        c0, c1 = rm_C[e]

                        # (dA, B otimes C)
                        # dA = ([dA00, dA01], [dA01, dA11])
                        # B = [b0, b1]
                        # C = [c0, c1]
                        # B otimes C = ([b0 * c0, b0 * c1], [b1 * c0, b1 * c1])
                        # dA.dot( B otimes C ) = (
                        #       [dA00 * b0 * c0, dA01 * b0 * c1],
                        #       [dA10 * b1 * c0, dA11 * b1 * c1],
                        # )

                        metric = quad_weights * dJi

                        o00 = np.einsum(
                            'li, lj, lk, l -> ijk', dA00, b0, c0, metric, optimize='optimal'
                        )
                        o01 = np.einsum(
                            'li, lj, lk, l -> ijk', dA01, b0, c1, metric, optimize='optimal'
                        )
                        o10 = np.einsum(
                            'li, lj, lk, l -> ijk', dA10, b1, c0, metric, optimize='optimal'
                        )
                        o11 = np.einsum(
                            'li, lj, lk, l -> ijk', dA11, b1, c1, metric, optimize='optimal'
                        )

                        data = o00 + o01 + o10 + o11

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

        if dimensions == 1:
            raise NotImplementedError()

        elif dimensions == 2:

            row_form, col_form = args

            return self._2d_matrix_representation(row_form, col_form)

        elif dimensions == 3:
            raise NotImplementedError()

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
            caller = _D2MatrixCallerRow2Col1(self._ABC, self._3d_data)

        elif row_index == 2 and col_index == 0:
            # caller = self._2d_matrix_caller_r2_c0
            caller = _D2MatrixCallerRow2Col0(self._ABC, self._3d_data)

        else:
            raise NotImplementedError()

        return MsePyDynamicLocalMatrix(caller)
