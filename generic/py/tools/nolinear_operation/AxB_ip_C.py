# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.quadrature import Quadrature
from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from generic.py.tools.nolinear_operation.base_3_entries import Base3Entries

_3d_data_cache_py_AxB_C = {}


class _AxBipC(Base3Entries):
    """"""

    def __init__(self, A, B, C):
        """(AxB, C)"""
        super().__init__()
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        self._mesh = A.mesh
        self._A = A
        self._B = B
        self._C = C
        self._ABC = (A, B, C)
        if all([_.space.abstract.__class__ is ScalarValuedFormSpace for _ in self._ABC]):
            self._type = 'Lambda'
        else:
            raise NotImplementedError()
        cache_key = list()
        for f in (A, B, C):
            cache_key.append(
                f.space.__repr__() + '@degree:' + _degree_str_maker(f.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._cache_key = cache_key
        self._3d_data = None
        self._freeze()

    def _make_3d_data(self):
        """"""
        if self._3d_data is not None:
            return
        else:
            pass

        if self._cache_key in _3d_data_cache_py_AxB_C:
            self._3d_data = _3d_data_cache_py_AxB_C[self._cache_key]
            return

        mesh = self._mesh
        n = mesh.n

        degrees = list()
        for form in self._ABC:
            p = form.space[form._degree].p
            if isinstance(p, int):
                p = [p for _ in range(n)]
            else:
                pass
            degrees.append(p)

        degrees = np.array(degrees)
        degrees = np.max(degrees, axis=0)
        degrees = [int(_ * 1.5) + 1 for _ in degrees]

        quad = Quadrature(degrees, category='Gauss')
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rmA = self._A.reconstruction_matrix(*quad_nodes)
        rmB = self._B.reconstruction_matrix(*quad_nodes)
        rmC = self._C.reconstruction_matrix(*quad_nodes)

        csm_A = self._A.space.basis_functions.csm(self._A.degree)
        csm_B = self._B.space.basis_functions.csm(self._B.degree)
        csm_C = self._C.space.basis_functions.csm(self._C.degree)

        if n == 2:
            xi, et = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            detJ = mesh.ct.Jacobian(xi, et)
        elif n == 3:
            xi, et, sg = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            sg = sg.ravel('F')
            detJ = mesh.ct.Jacobian(xi, et, sg)
        else:
            raise Exception()

        _data_cache = dict()
        _3d_data = dict()

        for index in mesh:
            # --------------------------------------------------------------------------
            if index in csm_A or index in csm_B or index in csm_C:
                cache_index = index
            else:
                cache_index = mesh[index].metric_signature

            # --------------------------------------------------------------------------
            if cache_index in _data_cache:
                _3d_data[index] = _data_cache[cache_index]
            else:

                if len(rmA[index]) == 1 and len(rmB[index]) == len(rmC[index]) == 2:
                    # A is a 0-form, B, C are 1-forms!
                    # so, A = [0 0 w]^T, B = [u, v, 0]^T, C = [a b 0]^T,
                    # cp_term = A X B = [-wv wu 0]^T, (cp_term, C) = -wva + wub
                    w = rmA[index][0]
                    u, v = rmB[index]
                    a, b = rmC[index]
                    dJi = detJ[index]
                    data = - np.einsum(
                        'li, lj, lk, l -> ijk', w, v, a, quad_weights * dJi, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', w, u, b, quad_weights * dJi, optimize='optimal'
                    )
                else:
                    raise NotImplementedError()

                _3d_data[index] = data

                _data_cache[cache_index] = data

            # =================================================================
        self._3d_data = _3d_data
        _3d_data_cache_py_AxB_C[self._cache_key] = _3d_data
