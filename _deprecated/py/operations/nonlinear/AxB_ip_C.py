# -*- coding: utf-8 -*-
r"""
"""
import numpy as np
from src.spaces.main import _degree_str_maker
from tools.quadrature import Quadrature
from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from msehy.py.operations.nonlinear.base_3_entries import IrregularBase3Entries

_irregular_3d_data_cache = dict()


class _AxBipC(IrregularBase3Entries):
    """"""

    def __init__(self, A, B, C, quad=None):
        """(AxB, C)"""
        super().__init__()
        assert A.mesh is B.mesh and A.mesh is C.mesh, f"Meshes do not match!"
        cache_key = list()
        for form in (A, B, C):
            assert hasattr(form, '_is_discrete_form') and form._is_discrete_form(), \
                f"{form} is not a discrete form!"
            cache_key.append(
                form.space.__repr__() + '@degree:' + _degree_str_maker(form.degree)
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
        self._mesh = A.mesh
        self._freeze()

    def _3d_data(self, g):
        """Make the 3d data at generation g."""
        g = self._mesh._pg(g)

        cached, _3d_data = self._mesh.generations.sync_cache(
            _irregular_3d_data_cache, g, self._cache_key)
        if cached:
            return _3d_data
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
        else:
            raise NotImplementedError()

        quad_degrees, quad_types = degrees, types

        quad = Quadrature(quad_degrees, category=types)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rmA = self._A.reconstruction_matrix(*quad_nodes, g=g)
        rmB = self._B.reconstruction_matrix(*quad_nodes, g=g)
        rmC = self._C.reconstruction_matrix(*quad_nodes, g=g)

        representative = self._mesh[g]

        if self._mesh.n == 2:
            xi, et = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            detJ = representative.ct.Jacobian(xi, et)
        elif self._mesh.n == 3:
            xi, et, sg = np.meshgrid(*quad_nodes, indexing='ij')
            xi = xi.ravel('F')
            et = et.ravel('F')
            sg = sg.ravel('F')
            detJ = representative.ct.Jacobian(xi, et, sg)
        else:
            raise Exception()

        _data_cache = dict()
        _3d_data = dict()

        for i in representative:

            cache_index = representative[i].metric_signature

            if cache_index in _data_cache:
                pass

            else:
                if self._type == 'scalar':
                    # make the data --------------- for 2d meshes -----------------------------------
                    if self._mesh.n == 2:
                        if len(rmA[i]) == 1 and len(rmB[i]) == len(rmC[i]) == 2:
                            # A is a 0-form, B, C are 1-forms!
                            # so, A = [0 0 w]^T, B = [u, v, 0]^T, C = [a b 0]^T,
                            # cp_term = A X B = [-wv wu 0]^T, (cp_term, C) = -wva + wub
                            w = rmA[i][0]
                            u, v = rmB[i]
                            a, b = rmC[i]
                            dJi = detJ[i]
                            data = - np.einsum(
                                'li, lj, lk, l -> ijk', w, v, a, quad_weights * dJi, optimize='optimal'
                            ) + np.einsum(
                                'li, lj, lk, l -> ijk', w, u, b, quad_weights * dJi, optimize='optimal'
                            )
                        elif len(rmA[i]) == len(rmB[i]) == 2 and len(rmC[i]) == 1:
                            # A, B are 1-forms, C is a 0-form!
                            # so, A = [wx wy, 0]^T    B = [u v 0]^T   C= [0 0 c]^T
                            # A x B = [wy*0 - 0*v,   0*u - wx*0,   wx*v - wy*u]^T = [0 0 C0]^T
                            # (A x B) dot C = 0*0 + 0*0 + C0*c = wx*v*c - wy*u*c
                            wx, wy = rmA[i]
                            u, v = rmB[i]
                            c = rmC[i][0]
                            dJi = detJ[i]
                            data = - np.einsum(
                                'li, lj, lk, l -> ijk', wx, v, c, quad_weights * dJi, optimize='optimal'
                            ) + np.einsum(
                                'li, lj, lk, l -> ijk', wy, u, c, quad_weights * dJi, optimize='optimal'
                            )

                        else:
                            raise NotImplementedError()

                    # make the data --------------- for 3d meshes -----------------------------------
                    elif self._mesh.n == 3:
                        if len(rmA[i]) == len(rmB[i]) == len(rmC[i]) == 3:
                            # A, B, C are all vectors.
                            # A = [wx wy, wz]^T    B = [u v w]^T   C= [a b c]^T
                            # A x B = [wy*w - wz*v,   wz*u - wx*w,   wx*v - wy*u]^T = [A0 B0 C0]^T
                            # (A x B) dot C = A0*a + B0*b + C0*c
                            wx, wy, wz = rmA[i]
                            u, v, w = rmB[i]
                            a, b, c = rmC[i]
                            dJi = detJ[i]
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

            _3d_data[i] = _data_cache[cache_index]

        self._mesh.generations.sync_cache(
            _irregular_3d_data_cache, g, self._cache_key, data=_3d_data)

        return _3d_data
