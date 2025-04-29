# -*- coding: utf-8 -*-
"""
"""

from tools.frozen import Frozen
from msehtt.static.space.inner_product.Lambda_Lambda.IP_LL_m2n2k1 import ___Lambda_ip_Lambda_over_SameTPM_m2n2k1___


class MseHttSpace_Lambda_ip_Lambda(Frozen):
    """"""
    def __init__(self):
        r""""""
        self._freeze()

    def __call__(
        self,
        self_space, self_degree, self_cochain,
        other_space, other_degree, other_cochain,
        inner_type='L2'
    ):
        r""""""
        assert self_space.tpm is other_space.tpm, f"works only for a same grid."
        s_mn, s_k = self_space.mn, self_space.abstract.k
        o_mn, o_k = other_space.mn, other_space.abstract.k

        if s_mn == o_mn == (2, 2):
            if s_k == o_k == 1:
                return ___Lambda_ip_Lambda_over_SameTPM_m2n2k1___(
                    self_space, self_degree, self_cochain,
                    other_space, other_degree, other_cochain,
                    inner_type=inner_type
                )
            else:
                raise NotImplementedError()

        else:
            raise NotImplementedError()
