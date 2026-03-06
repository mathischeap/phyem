# -*- coding: utf-8 -*-
r"""
"""

from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_base import MseHttTrilinearBase
from phyem.msehtt.static.form.main import MseHttForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.tools.quadrature import quadrature

import numpy as np

___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___ = {}


class AdB_ip_C(MseHttTrilinearBase):
    r"""(A * dB, C)"""
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
        cache_key = ' <A_dB__ip__C> '.join(cache_key)
        self._melt()
        self._cache_key = cache_key
        self._tpm = A.tpm
        self._tgm = A.tgm
        self._freeze()

    @classmethod
    def clean_cache(cls):
        r""""""
        keys = list(___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___.keys())
        for key in keys:
            del ___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___[key]

    def _make_3d_data(self):
        """"""
        # ---- if the data is already there -------------------------------------------
        if self._3d_data is not None:
            pass
        # ---- if the data is cached ---------------------------------------------------
        elif self._cache_key in ___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___:
            self._3d_data = ___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___[self._cache_key]
        # ------- make the data -------------------------------------------------------------
        else:
            _3d_data = self._generate_data_()
            self._3d_data = _3d_data
            ___msehtt_static_TriLinearCache_AdB_ip_C_3d_data___[self._cache_key] = _3d_data

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

        EB = self._B.incidence_matrix
        dB = self._B.d()
        rm_db = dB.reconstruction_matrix(*quad_nodes)
        rmB = [{} for _ in range(len(rm_db))]
        for i in EB:
            ei = EB[i]
            for j in range(len(rm_db)):
                rmb = rm_db[j][i]
                rmB[j][i] = rmb @ ei

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

                # Remember, d of B has been included into reconstructing matrix of B, i.e. rmB.

                # So, we only need to consider it like `int_{(A * B, C)}`, i.e. integral of (A multi B, C).
                # And this `multi` can be `dot-product` if it is for vectors.
                if indicator == 'm2n2=(1, 2, 2)':  # on 2d manifold in 2d space, A is scalar, B, C are vector.
                    # <A B, C>; int{A, B, C}; int{A B · C}
                    w = rmA[0][e]
                    u, v = rmB[0][e], rmB[1][e]
                    a, b = rmC[0][e], rmC[1][e]
                    # int{w(ua + vb)} = int{wua + wvb}
                    element_3d_data = np.einsum(
                        'li, lj, lk, l -> ijk', w, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', w, v, b, qw_ravel * detJ, optimize='optimal'
                    )

                elif indicator == 'm2n2=(1, 1, 1)':  # on 2d manifold in 2d space, A, B, C are all scalars.
                    # int{A * B * C}
                    a = rmA[0][e]
                    b = rmB[0][e]
                    c = rmC[0][e]
                    # int{abc}
                    element_3d_data = np.einsum(
                        'li, lj, lk, l -> ijk', a, b, c, qw_ravel * detJ, optimize='optimal'
                    )

                elif indicator == 'm3n3=(1, 3, 3)':  # on 3d manifold in 3d space, A is scalar, B, C are vector.
                    # <A B, C>; int{A, B, C}; int{A B · C}
                    w = rmA[0][e]
                    u, v, W = rmB[0][e], rmB[1][e], rmB[2][e]
                    a, b, c = rmC[0][e], rmC[1][e], rmC[2][e]
                    # int{w(ua + vb)} = int{wua + wvb}
                    element_3d_data = np.einsum(
                        'li, lj, lk, l -> ijk', w, u, a, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', w, v, b, qw_ravel * detJ, optimize='optimal'
                    ) + np.einsum(
                        'li, lj, lk, l -> ijk', w, W, c, qw_ravel * detJ, optimize='optimal'
                    )

                elif indicator == 'm3n3=(1, 1, 1)':  # on 3d manifold in 3d space, A, B, C are all scalars.
                    # int{A * B * C}
                    a = rmA[0][e]
                    b = rmB[0][e]
                    c = rmC[0][e]
                    # int{abc}
                    element_3d_data = np.einsum(
                        'li, lj, lk, l -> ijk', a, b, c, qw_ravel * detJ, optimize='optimal'
                    )

                else:
                    raise NotImplementedError(
                        f"indicator={indicator} is not coded for trilinear {self.__class__.__name__}."
                    )

                _3d_data[e] = element_3d_data
                if isinstance(cache_key, str):
                    _cache_[cache_key] = element_3d_data
                else:
                    pass

        return _3d_data
