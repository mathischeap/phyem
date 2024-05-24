# -*- coding: utf-8 -*-
"""
"""

from msehtt.static.form.addons.nop_data_computer.trilinear_base import MseHttTrilinearBase
from msehtt.static.form.main import MseHttForm
from src.spaces.main import _degree_str_maker
from tools.quadrature import quadrature
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
        if isinstance(self._A.degree, (float, int)):
            quad_degree_A = self._A.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-A degree = {self._A.degree}")
        if isinstance(self._B.degree, (float, int)):
            quad_degree_B = self._B.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-B degree = {self._B.degree}")
        if isinstance(self._C.degree, (float, int)):
            quad_degree_C = self._C.degree
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-C degree = {self._C.degree}")

        quad_degree = int(max([quad_degree_A, quad_degree_B, quad_degree_C]) * 1.5) + 1

        if self._tpm.abstract.m == self._tpm.abstract.n == 2:
            indicator = 'm2n2'
            quad = quadrature((quad_degree, quad_degree), category='Gauss')
            quad_nodes = quad.quad_nodes
            qw_ravel = quad.quad_weights_ravel
        else:
            raise NotImplementedError()
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
            metric_signature = element.metric_signature
            if isinstance(metric_signature, str) and metric_signature in _cache_:
                _3d_data[e] = _cache_[metric_signature]
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

                else:
                    raise NotImplementedError(indicator)

                _3d_data[e] = element_3d_data
                if isinstance(metric_signature, str):
                    _cache_[metric_signature] = element_3d_data
                else:
                    pass

        return _3d_data
