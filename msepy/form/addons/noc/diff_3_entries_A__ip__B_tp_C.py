# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker
from tools.quadrature import Quadrature
from src.spaces.continuous.bundle import BundleValuedFormSpace
from msepy.form.addons.noc.diff_3_entries__BASE import Base3Entries

_3d_data_cache_2 = {}


# noinspection PyPep8Naming
class _A__ip__B_tp_C_(Base3Entries):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(A, B otimes C)"""
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

        if self._cache_key in _3d_data_cache_2:
            self._3d_data = _3d_data_cache_2[self._cache_key]
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

        rm_A = self._A.reconstruction_matrix(*quad_nodes)
        rm_B = self._B.reconstruction_matrix(*quad_nodes)
        rm_C = self._C.reconstruction_matrix(*quad_nodes)

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
                    # make the data --------------- for 2d meshes ---------------------------
                    if self._mesh.n == 2:

                        dJi = detJ[e]
                        metric = quad_weights * dJi

                        A0, A1 = rm_A[e]
                        A00, A01 = A0
                        A10, A11 = A1

                        b0, b1 = rm_B[e]
                        c0, c1 = rm_C[e]

                        # (A, B otimes C)
                        # A = ([A00, A01], [A01, A11])
                        # B = [b0, b1]
                        # C = [c0, c1]
                        # B otimes C = ([b0 * c0, b0 * c1], [b1 * c0, b1 * c1])
                        # A.dot( B otimes C ) = (
                        #       [A00 * b0 * c0, A01 * b0 * c1],
                        #       [A10 * b1 * c0, A11 * b1 * c1],
                        # )

                        o00 = np.einsum(
                            'li, lj, lk, l -> ijk', A00, b0, c0, metric, optimize='optimal'
                        )
                        o01 = np.einsum(
                            'li, lj, lk, l -> ijk', A01, b0, c1, metric, optimize='optimal'
                        )
                        o10 = np.einsum(
                            'li, lj, lk, l -> ijk', A10, b1, c0, metric, optimize='optimal'
                        )
                        o11 = np.einsum(
                            'li, lj, lk, l -> ijk', A11, b1, c1, metric, optimize='optimal'
                        )

                        data = o00 + o01 + o10 + o11

                    # else: must be wrong, we do not do this in 1d -------------------------
                    else:
                        raise Exception()

                else:
                    raise NotImplementedError()

                _data_cache[cache_index] = data

            _3d_data[e] = _data_cache[cache_index]

        self._3d_data = _3d_data
        _3d_data_cache_2[self._cache_key] = _3d_data
