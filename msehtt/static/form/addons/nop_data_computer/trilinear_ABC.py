# -*- coding: utf-8 -*-
r"""
"""

from phyem.msehtt.static.form.addons.nop_data_computer.trilinear_base import MseHttTrilinearBase
from phyem.msehtt.static.form.main import MseHttForm
from phyem.src.spaces.main import _degree_str_maker
from phyem.tools.quadrature import quadrature

import numpy as np

_cache_ABC_3d_data_ = {}


class T_ABC(MseHttTrilinearBase):
    """integral of ABC over the domain."""
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
        cache_key = ' <*> '.join(cache_key)
        self._melt()
        self._cache_key = cache_key
        self._tpm = A.tpm
        self._tgm = A.tgm
        self._freeze()

    @classmethod
    def clean_cache(cls):
        r""""""
        keys = list(_cache_ABC_3d_data_.keys())
        for key in keys:
            del _cache_ABC_3d_data_[key]

    def _make_3d_data(self):
        """"""
        # ---- if the data is already there -------------------------------------------
        if self._3d_data is not None:
            pass
        # ---- if the data is cached ---------------------------------------------------
        elif self._cache_key in _cache_ABC_3d_data_:
            self._3d_data = _cache_ABC_3d_data_[self._cache_key]
        # ------- make the data -------------------------------------------------------------
        else:
            _3d_data = self._generate_data_()
            self._3d_data = _3d_data
            _cache_ABC_3d_data_[self._cache_key] = _3d_data

    def _generate_data_(self):
        """"""
        if isinstance(self._A.degree, (float, int)):
            quad_degree_A = self._A.degree
        elif isinstance(self._A.degree, (list, tuple)) and all([isinstance(_, int) for _ in self._A.degree]):
            quad_degree_A = max(self._A.degree)
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-A degree = {self._A.degree}")
        if isinstance(self._B.degree, (float, int)):
            quad_degree_B = self._B.degree
        elif isinstance(self._B.degree, (list, tuple)) and all([isinstance(_, int) for _ in self._B.degree]):
            quad_degree_B = max(self._B.degree)
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-B degree = {self._B.degree}")
        if isinstance(self._C.degree, (float, int)):
            quad_degree_C = self._C.degree
        elif isinstance(self._C.degree, (list, tuple)) and all([isinstance(_, int) for _ in self._C.degree]):
            quad_degree_C = max(self._C.degree)
        else:
            raise NotImplementedError(f"cannot find a quad degree from form-C degree = {self._C.degree}")

        quad_degree = int(max([quad_degree_A, quad_degree_B, quad_degree_C]) * 1.5) + 1

        if self._tpm.abstract.m == self._tpm.abstract.n == 2:
            indicator = 'm2n2'
            quad = quadrature((quad_degree, quad_degree), category='Gauss')
        elif self._tpm.abstract.m == self._tpm.abstract.n == 3:
            indicator = 'm3n3'
            quad = quadrature((quad_degree, quad_degree, quad_degree), category='Gauss')
        else:
            raise NotImplementedError()

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
            metric_signature = element.metric_signature
            etype = element.etype

            if etype in (
                    "orthogonal rectangle",
                    "unique msepy curvilinear quadrilateral",
                    "orthogonal hexahedron",
            ):
                cache_key = metric_signature

            elif etype == 9:
                reverse_info = element.dof_reverse_info
                if 'm2n2k1_outer' in reverse_info:
                    reverse_key_outer = str(reverse_info['m2n2k1_outer'])
                else:
                    reverse_key_outer = ''
                if 'm2n2k1_inner' in reverse_info:
                    reverse_key_inner = str(reverse_info['m2n2k1_inner'])
                else:
                    reverse_key_inner = ''
                cache_key = metric_signature + '-' + reverse_key_outer + ':' + reverse_key_inner

            elif etype in (
                "unique msepy curvilinear hexahedron",
            ):
                cache_key = None

            else:
                raise NotImplementedError(f"{self.__class__.__name__} generate_data not implemented for e_type={etype}")

            if isinstance(cache_key, str) and cache_key in _cache_:
                _3d_data[e] = _cache_[cache_key]

            else:
                detJ = element.ct.Jacobian(*metric_coo)
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

                else:
                    raise NotImplementedError(f"indicator={indicator} is not coded for trilinear ABC.")

                _3d_data[e] = element_3d_data
                if isinstance(cache_key, str):
                    _cache_[cache_key] = element_3d_data
                else:
                    pass

        return _3d_data
