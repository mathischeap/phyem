# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from msepy.form.main import MsePyRootForm
from src.spaces.main import _degree_str_maker
from tools.quadrature import Quadrature
from src.spaces.continuous.Lambda import ScalarValuedFormSpace
from msepy.form.addons.noc._4_entries__BASE import Base4Entries

_AxB_ip_CxD__data_cache_ = {}


class AxB_ip_CxD(Base4Entries):
    """"""
    def __init__(self, A, B, C, D):
        """(AxB, C)"""
        super().__init__()
        assert A.mesh is B.mesh and A.mesh is C.mesh and A.mesh is D.mesh, f"Meshes do not match!"
        cache_key = list()
        for msepy_form in (A, B, C, D):
            assert msepy_form.__class__ is MsePyRootForm, f"{msepy_form} is not a {MsePyRootForm}!"
            cache_key.append(
                msepy_form.space.__repr__() + '@degree:' + _degree_str_maker(msepy_form.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._cache_key = cache_key
        self._A = A
        self._B = B
        self._C = C
        self._D = D
        self._ABCD = (A, B, C, D)
        if all([_.space.abstract.__class__ is ScalarValuedFormSpace for _ in self._ABCD]):
            self._type = 'scalar'
        else:
            raise NotImplementedError()
        self._4d_data = None
        self._mesh = A.mesh
        self._e2c = A.mesh.elements._index_mapping._e2c

    def _make_4d_data(self):
        """"""
        if self._4d_data is not None:
            return
        else:
            pass

        if self._cache_key in _AxB_ip_CxD__data_cache_:
            self._4d_data = _AxB_ip_CxD__data_cache_[self._cache_key]
            return

        degrees = list()
        for form in self._ABCD:
            degrees.append(
                form.space[form._degree].p
            )
        degrees = np.array(degrees)
        degrees = np.max(degrees, axis=0)
        degrees = [int(_ * 2) + 1 for _ in degrees]
        types = 'Gauss'

        quad_degrees, quad_types = degrees, types

        quad = Quadrature(quad_degrees, category=types)
        quad_nodes = quad.quad_nodes
        quad_weights = quad.quad_weights_ravel

        rmA = self._A.reconstruction_matrix(*quad_nodes)
        rmB = self._B.reconstruction_matrix(*quad_nodes)
        rmC = self._C.reconstruction_matrix(*quad_nodes)
        rmD = self._D.reconstruction_matrix(*quad_nodes)

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
        _4d_data = dict()

        for e in self._mesh.elements:

            cache_index = self._e2c[e]

            if cache_index in _data_cache:
                pass

            else:
                if self._type == 'scalar':
                    if self._mesh.n == 2:
                        if len(rmA[e]) == len(rmB[e]) == len(rmC[e]) == len(rmD[e]) == 2:
                            # A, B, C, D are all 1-forms.
                            # let A = [wx wy 0]^T    B = [u v 0]^T
                            # A x B = [wy*0 - 0*v   0*u - wx*0   wx*v - wy*u]^T = [0   0   C0]^T
                            # let C = [cx cy 0]^T    D = [U V 0]^T
                            # C x D = [cy*0 - 0*v   0*u - cx*0   cx*V - cy*U]^T = [0   0   D0]^T
                            # (A x B) dot (C x D) = C0 * D0 = (wx*v - wy*u) * (cx*V - cy*U)
                            #                               = wx*v*cx*V - wx*v*cy*U - wy*u*cx*V + wy*u*cy*U
                            wx, wy = rmA[e]
                            u, v = rmB[e]
                            cx, cy = rmC[e]
                            U, V = rmD[e]

                            dJi = detJ[e]

                            data = np.einsum(
                                'li, lj, lk, lm, l -> ijkm', wx, v, cx, V, quad_weights * dJi, optimize='optimal'
                            ) - np.einsum(
                                'li, lj, lk, lm, l -> ijkm', wx, v, cy, U, quad_weights * dJi, optimize='optimal'
                            ) - np.einsum(
                                'li, lj, lk, lm, l -> ijkm', wy, u, cx, V, quad_weights * dJi, optimize='optimal'
                            ) + np.einsum(
                                'li, lj, lk, lm, l -> ijkm', wy, u, cy, U, quad_weights * dJi, optimize='optimal'
                            )

                        else:
                            raise NotImplementedError()

                    elif self._mesh.n == 3:
                        raise NotImplementedError()
                    else:
                        raise Exception()
                else:
                    raise NotImplementedError()

                _data_cache[cache_index] = data

            _4d_data[e] = _data_cache[cache_index]

        self._4d_data = _4d_data
        _AxB_ip_CxD__data_cache_[self._cache_key] = _4d_data
