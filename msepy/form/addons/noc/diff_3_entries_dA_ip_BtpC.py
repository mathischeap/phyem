# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker
from src.spaces.continuous.bundle import BundleValuedFormSpace
from tools.quadrature import Quadrature
from msepy.form.addons.noc.diff_3_entries__BASE import Base3Entries

_3d_data_cache_1 = {}


# noinspection PyPep8Naming
class __dA_ip_BtpC__(Base3Entries):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(AxB, C)"""
        super().__init__()
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        cache_key = list()
        for msepy_form in (A, B, C):
            assert msepy_form.__class__ is MsePyRootForm, f"{msepy_form} is not a {MsePyRootForm}!"
            cache_key.append(
                msepy_form.space.__repr__() + '@degree:' + _degree_str_maker(msepy_form.degree)
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
        self._mesh = A.mesh
        self._e2c = A.mesh.elements._index_mapping._e2c
        self._freeze()

    def _make_3d_data(self):
        """"""

        if self._3d_data is not None:
            return
        else:
            pass

        if self._cache_key in _3d_data_cache_1:
            self._3d_data = _3d_data_cache_1[self._cache_key]
            return

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
        _3d_data_cache_1[self._cache_key] = _3d_data
