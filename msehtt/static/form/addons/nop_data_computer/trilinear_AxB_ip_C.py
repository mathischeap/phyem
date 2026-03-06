# -*- coding: utf-8 -*-
r"""
"""

from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_base import MseHttTrilinearBase
from phyem.msehtt.static.form.main import MseHttForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.tools.quadrature import quadrature

import numpy as np

_cache_AxB_ip_C_3d_data_ = {}


class AxB_ip_C(MseHttTrilinearBase):
    """"""
    def __init__(self, A, B, C):
        super().__init__(A, B, C)
        assert A.tgm is B.tgm and A.tgm is C.tgm, f"the great meshes do not match!"
        assert A.tpm is B.tpm and A.tpm is C.tpm, f"the partial meshes do not match!"
        cache_key = list()
        for f in (A, B, C):
            assert f.__class__ is MseHttForm, f"{f} is not a {MseHttForm}!"
            cache_key.append(
                f.space.__repr__() + '@degree:' + _degree_str_maker(f.degree)
            )
        cache_key = ' <=> '.join(cache_key)
        self._melt()
        self._cache_key = cache_key
        self._tpm = A.tpm
        self._tgm = A.tgm
        self._freeze()

    @classmethod
    def clean_cache(cls):
        r""""""
        keys = list(_cache_AxB_ip_C_3d_data_.keys())
        for key in keys:
            del _cache_AxB_ip_C_3d_data_[key]

    def _make_3d_data(self):
        """"""
        # ---- if the data is already there -------------------------------------------
        if self._3d_data is not None:
            pass
        # ---- if the data is cached ---------------------------------------------------
        elif self._cache_key in _cache_AxB_ip_C_3d_data_:
            self._3d_data = _cache_AxB_ip_C_3d_data_[self._cache_key]
        # ------- make the data -------------------------------------------------------------
        else:
            _3d_data = self._generate_data_()
            self._3d_data = _3d_data
            _cache_AxB_ip_C_3d_data_[self._cache_key] = _3d_data

    def _generate_data_(self):
        """"""
        from phyem.msehtt.static.mesh.great.elements.types.base import MseHttGreatMeshBaseElement
        pA, _ = MseHttGreatMeshBaseElement.degree_parser(self._A.degree, m=self._tpm.abstract.m, n=self._tpm.abstract.n)
        pB, _ = MseHttGreatMeshBaseElement.degree_parser(self._B.degree, m=self._tpm.abstract.m, n=self._tpm.abstract.n)
        pC, _ = MseHttGreatMeshBaseElement.degree_parser(self._C.degree, m=self._tpm.abstract.m, n=self._tpm.abstract.n)
        quad_p = int(max([max(pA), max(pB), max(pC)]) * 1.5) + 1
        if self._tpm.abstract.m == self._tpm.abstract.n == 2:
            indicator = 'm2n2'
            quad_p = (quad_p, quad_p)
        elif self._tpm.abstract.m == self._tpm.abstract.n == 3:
            indicator = 'm3n3'
            quad_p = (quad_p, quad_p, quad_p)
        else:
            raise NotImplementedError()
        quad = quadrature(quad_p, category='Gauss')

        quad_nodes = quad.quad_nodes
        qw_ravel = quad.quad_weights_ravel

        metric_coo = [_.ravel('F') for _ in np.meshgrid(*quad_nodes, indexing='ij')]

        rmA = self._A.reconstruction_matrix(*quad_nodes)
        rmB = self._B.reconstruction_matrix(*quad_nodes)
        rmC = self._C.reconstruction_matrix(*quad_nodes)

        indicator += '=' + str((len(rmA), len(rmB), len(rmC)))

        _cache_ = {}
        _3d_data = {}
        elements = self._tpm.composition
        for e in elements:
            element = elements[e]
            cache_key = element.metric_bf_cache_key()

            if isinstance(cache_key, str) and cache_key in _cache_:
                _3d_data[e] = _cache_[cache_key]

            else:
                detJ = element.ct.Jacobian(*metric_coo)
                if indicator == 'm2n2=(1, 2, 2)':  # on 2d manifold in 2d space, A is scalar, B, C are vector.
                    # A is a 0-form, B, C are 1-forms!
                    # so, A = [0 0 w]^T, B = [u v 0]^T, C = [a b 0]^T,
                    # cp_term = A X B = [-wv wu 0]^T, (cp_term, C) = -wva + wub
                    w = rmA[0][e]
                    u, v = rmB[0][e], rmB[1][e]
                    a, b = rmC[0][e], rmC[1][e]
                    element_3d_data = - np.einsum(
                        'li, lj, lk, l -> ijk', w, v, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', w, u, b, qw_ravel * detJ, optimize='optimal'
                    )

                elif indicator == 'm2n2=(2, 2, 1)':  # on 2d manifold in 2d space, A, B are vectors, C is scalar.
                    # A, B are 1-forms, C is a 0 (or 2)-form!
                    # so, A = [wx wy 0]^T    B = [u v 0]^T   C= [0 0 c]^T
                    # A x B = [wy*0 - 0*v   0*u - wx*0   wx*v - wy*u]^T = [0 0 C0]^T
                    # (A x B) dot C = 0*0 + 0*0 + C0*c = wx*v*c - wy*u*c
                    wx, wy = rmA[0][e], rmA[1][e]
                    u, v = rmB[0][e], rmB[1][e]
                    c = rmC[0][e]
                    element_3d_data = np.einsum(
                        'li, lj, lk, l -> ijk', wx, v, c, qw_ravel * detJ, optimize='optimal'
                    ) - np.einsum(
                        'li, lj, lk, l -> ijk', wy, u, c, qw_ravel * detJ, optimize='optimal'
                    )

                elif indicator == 'm3n3=(3, 3, 3)':
                    # A, B, C are all vectors.
                    # A = [wx wy, wz]^T    B = [u v w]^T   C= [a b c]^T
                    # A x B = [wy*w - wz*v   wz*u - wx*w   wx*v - wy*u]^T = [A0 B0 C0]^T
                    # (A x B) dot C = A0*a + B0*b + C0*c
                    wx, wy, wz = rmA[0][e], rmA[1][e], rmA[2][e]
                    u, v, w = rmB[0][e], rmB[1][e], rmB[2][e]
                    a, b, c = rmC[0][e], rmC[1][e], rmC[2][e]
                    A0a = np.einsum(
                        'li, lj, lk, l -> ijk', wy, w, a, qw_ravel * detJ, optimize='optimal'
                    ) - np.einsum(
                        'li, lj, lk, l -> ijk', wz, v, a, qw_ravel * detJ, optimize='optimal'
                    )
                    B0b = np.einsum(
                        'li, lj, lk, l -> ijk', wz, u, b, qw_ravel * detJ, optimize='optimal'
                    ) - np.einsum(
                        'li, lj, lk, l -> ijk', wx, w, b, qw_ravel * detJ, optimize='optimal'
                    )
                    C0c = np.einsum(
                        'li, lj, lk, l -> ijk', wx, v, c, qw_ravel * detJ, optimize='optimal'
                    ) - np.einsum(
                        'li, lj, lk, l -> ijk', wy, u, c, qw_ravel * detJ, optimize='optimal'
                    )

                    element_3d_data = A0a + B0b + C0c

                else:
                    raise NotImplementedError(indicator)

                _3d_data[e] = element_3d_data
                if isinstance(cache_key, str):
                    _cache_[cache_key] = element_3d_data
                else:
                    pass

        return _3d_data
